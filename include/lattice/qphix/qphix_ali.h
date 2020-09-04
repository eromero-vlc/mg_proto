/*
 * qphix_ali.h
 *
 *  Created on: July 11, 2020
 *      Author: Eloy Romero <eloy@cs.wm.edu>
 */

#ifndef INCLUDE_LATTICE_QPHIX_QPHIX_ALI_H_
#define INCLUDE_LATTICE_QPHIX_QPHIX_ALI_H_

#include "lattice/coarse/coarse_deflation.h" // computeDeflation
#include "lattice/coarse/coarse_op.h"
#include "lattice/coarse/coarse_transfer.h"                // CoarseTransfer
#include "lattice/coarse/invfgmres_coarse.h"               // UnprecFGMRESSolverCoarseWrapper
#include "lattice/coloring.h"                              // Coloring
#include "lattice/eigs_common.h"                           // EigsParams
#include "lattice/fine_qdpxx/mg_params_qdpxx.h"            // SetupParams
#include "lattice/linear_operator.h"
#include "lattice/qphix/mg_level_qphix.h"                  // QPhiXMultigridLevels
#include "lattice/qphix/qphix_eo_clover_linear_operator.h" // QPhiXWilsonCloverEOLinearOperatorF
#include "lattice/qphix/qphix_mgdeflation.h"               // MGDeflation
#include "lattice/qphix/qphix_transfer.h"                  // QPhiXTransfer
#include "lattice/qphix/qphix_types.h"                     // QPhiXSpinorF
#include "lattice/qphix/vcycle_recursive_qphix.h"          // VCycleRecursiveQPhiXEO2
#include <MG_config.h>
#include <cassert>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>

#ifdef MG_QMP_COMMS
#    include <qmp.h>
#endif

namespace MG {

    namespace GlobalComm {

#ifdef MG_QMP_COMMS
        void GlobalSum(double &array) { QMP_sum_double_array(&array, 1); }
#else
        void GlobalSum(double &array) {}
#endif
    } // namespace GlobalComm

    template <typename T> inline double sum(const std::vector<T> &v) {
        return std::accumulate(v.begin(), v.end(), 0.0);
    }

    template <typename T> class vector3d : public std::vector<T> {
    public:
        vector3d(unsigned int m, unsigned int n, unsigned int o)
            : std::vector<T>(m * n * o), _m(m), _n(n), _o(o) {}
        const T &operator()(unsigned int i, unsigned int j, unsigned int k) const {
            return std::vector<T>::operator[](i *_n *_o + j * _o + k);
        }
        T &operator()(unsigned int i, unsigned int j, unsigned int k) {
            return std::vector<T>::operator[](i *_n *_o + j * _o + k);
        }

    private:
        unsigned int _m, _n, _o;
    };

    /*
     * Solve a linear system using the Approximate Lattice Inverse as a preconditioner
     *
     * If K is the ALI preconditioner and P is an approximate projector on the lower part of A's
     * spectrum, then the linear system A*x = b is solved as x = y + A^{-1}*P*b where K*A*y =
     * K*(I-P)*b. The preconditioner K approximates the links of A^{-1} for near neighbor sites. The
     * approach is effective if |[(I-P)*A^{-1}]_ij| decays quickly as i and j are further apart
     * sites.
     *
     * The projector is built using multigrid deflation (see MGDeflation) and K is reconstructed
     * with probing based on coloring the graph lattice.
     */

    class ALIPrec : public ImplicitLinearSolver<QPhiXSpinor>,
                    public LinearSolver<QPhiXSpinorF>,
                    public AuxiliarySpinors<CoarseSpinor> {

        using AuxQ = AuxiliarySpinors<QPhiXSpinor>;
        using AuxQF = AuxiliarySpinors<QPhiXSpinorF>;
        using AuxC = AuxiliarySpinors<CoarseSpinor>;

    public:
        /*
         * Constructor
         *
         * \param M_fine: linear system operator (A)
         * \param defl_p: Multigrid parameters used to build the multgrid deflation
         * \param defl_solver_params: linear system parameters to build the multigrid deflation
         * \param defl_eigs_params: eigensolver parameters to build the multigrid deflation
         * \param prec_p: Multigrid parameters used to build the preconditioner
         * \param K_distance: maximum distance of the approximated links
         * \param probing_distance: maximum distance for probing
         *
         * The parameters defl_p, defl_solver_params and defl_eigs_params are passed to MGDeflation
         * to build the projector P. The interesting values of (I-P)*A^{-1} are reconstructed with a
         * probing scheme that remove contributions from up to 'probing_distance' sites.
         *
         * _mode controls the projector Q on the preconditioner:
         *    for mode==0:
         *       Q = [M*\gamma_5*V(V^H*M*\gamma_5*V)^{-1}*V^H]_oo, where
         *       V is from _mg_deflation
         *
         *    for mode==1:
         *       Q = [L^{-1}*M*\gamma_5*V(V^H*M*\gamma_5*V)^{-1}*V^H*L]_oo
         *         = (M_oo-M_oe*M_ee^{-1}*M_eo)*[\gamma_5*V(V^H*M*\gamma_5*V)^{-1}*V^H]_oo, where
         *       V is from _mg_deflation
         *
         *    for mode==2:
         *       Q = [L^{-1}*M*V(V^H*M*V)^{-1}*V^H*L]_oo
         *         = (M_oo-M_oe*M_ee^{-1}*M_eo)*[V(V^H*M*V)^{-1}*V^H]_oo, where
         *       V is the whole prolongator in _mg_levels.
         *
         * Being K \approx (M_oo-M_oe*M_ee^{-1}*M_eo)^{-1}(I-Q) = A_oo^{-1}*(I-Q), _style controls
         * the output of operator():
         *    for style==0:
         *       out = K*in
         *
         *    for style==1:
         *       out = [A^{-1}*Q + K*(I-Q)]*in
         *
         *    for style==2:
         *       out = [A^{-1}*Q + (I-P)*K*(I-Q)]*in,
         *       where A^{-1}*(I-Q) == (I-P)*A^{-1}
         *
         *    for style==3:
         *       out = A^{-1}*Q*in + solve(A,(I-Q)*in,K*(I-Q)*in),
         *       where solve(A,b,x_0) approximately solves A*x=b using x_0 as initial guess.
         *
         *    for style==4: (only for _mode >= 2)
         *       out = A^{-1}*Q*in + solve(A,(I-Q)*in,(I-P)*K*(I-Q)*in),
         *       where solve(A,b,x_0) approximately solves A*x=b using x_0 as initial guess.
         *
         *    for style==5: (only for _mode >= 2) 
         *       out = A^{-1}*Q*in + solve(A,(I-Q)*in,K),
         *       where solve(A,b,K) approximately solves A*x=b using K as right preconditioner.
         *
         *    for style==6: (only for _mode >= 2)
         *       out = A^{-1}*Q*in + solve(A,(I-Q)*in,[A^{-1}*Q + (I-P)*K*(I-Q)]),
         *       where solve(A,b,K) approximately solves A*x=b using K as right preconditioner.
         *
         * prec_mode controls the form of the precoditioner:
         *    for prec_mode == 0:
         *       K = M_oo^{-1}
         *
         *    for prec_mode == 1
         *       K \approx M^{-1} \odot H*H^T,
         *       where HH^T is only nonzero on elements on the same site.
         */

        ALIPrec(const std::shared_ptr<const QPhiXWilsonCloverEOLinearOperatorF> M_fine,
                SetupParams defl_p, LinearSolverParamsBase defl_solver_params,
                EigsParams defl_eigs_params, SetupParams prec_p,
                std::vector<MG::VCycleParams> prec_vcycle_params,
                LinearSolverParamsBase prec_solver_params, unsigned int K_distance,
                unsigned int probing_distance, const CBSubset subset, unsigned int mode = 2,
                unsigned int style = 2, unsigned int prec_mode = 0)
            : ImplicitLinearSolver<QPhiXSpinor>(M_fine->GetInfo(), subset, prec_solver_params),
              LinearSolver<QPhiXSpinorF>(*M_fine, prec_solver_params),
              _M_fine(M_fine),
              _K_distance(K_distance),
              _subset(subset),
              _mode(mode),
              _style(style),
              _prec_mode(prec_mode) {
            (void)defl_p;

            MasterLog(INFO, "ALI Solver constructor: mode= %d style= %d  prec_mode= %d BEGIN",
                      _mode, _style, _prec_mode);

            // Create Multigrid preconditioner
            _mg_levels = std::make_shared<QPhiXMultigridLevelsEO>();
            SetupQPhiXMGLevels(prec_p, *_mg_levels, _M_fine);

            // Create stubs to self-calls
            _prec_coarse.resize(_mg_levels->coarse_levels.size());
            for (unsigned int level = 1; level < _mg_levels->coarse_levels.size() + 1; ++level) {
                _prec_coarse[level - 1] = std::make_shared<const S<CoarseSpinor>>(*this, level);
            }
            _prec_top = std::make_shared<const S<QPhiXSpinorF>>(*this);

            // Create a Multigrid V-cycle
            // Generate the prolongators and restrictiors
            assert(_mg_levels->coarse_levels.size() > 0);
            _Transfer_coarse_level.resize(_mg_levels->coarse_levels.size() - 1);
            for (int coarse_idx = (int)_mg_levels->coarse_levels.size() - 2; coarse_idx >= 0;
                 --coarse_idx) {
                _Transfer_coarse_level[coarse_idx] = std::make_shared<CoarseTransfer>(
                    _mg_levels->coarse_levels[coarse_idx].blocklist,
                    _mg_levels->coarse_levels[coarse_idx].null_vecs);
            }
            _Transfer_fine_level = std::make_shared<QPhiXTransfer<QPhiXSpinorF>>(
                _mg_levels->fine_level.blocklist, _mg_levels->fine_level.null_vecs);

            // Generate solvers for solving the whole coarse operator
            _bottom_solver.resize(_mg_levels->coarse_levels.size());
            for (unsigned int level = 1; level < _mg_levels->coarse_levels.size() + 1; ++level) {
                _bottom_solver[level - 1] = std::make_shared<UnprecFGMRESSolverCoarseWrapper>(
                    *_mg_levels->coarse_levels[level - 1].M,
                    prec_vcycle_params[level - 1].bottom_solver_params,
                    _prec_coarse[level - 1].get());
            }

            // Create projector
            _mg_deflation = std::make_shared<MGDeflation>(_M_fine, _mg_levels, defl_solver_params,
                                                          defl_eigs_params);

            // Set _op
            _op.resize(_mg_levels->coarse_levels.size() + 1);
            _op[0] = std::make_shared<CoarseDiracOp>(_M_fine->GetInfo());
            for (unsigned int level = 1; level < _mg_levels->coarse_levels.size() + 1; ++level)
                _op[level] =
                    std::make_shared<CoarseDiracOp>(*_mg_levels->coarse_levels[level - 1].info);

            // Build K
            _K_vals.resize(_mg_levels->coarse_levels.size() + 1);
            for (int level = _mg_levels->coarse_levels.size(); level >= 1; --level) {
                build_K(prec_solver_params, probing_distance,
                        *_mg_levels->coarse_levels[level - 1].M);
            }
            build_K(prec_solver_params, probing_distance, *_M_fine);
            MasterLog(INFO, "ALI Solver constructor: END");

            AuxQ::clear();
            AuxQF::clear();
            AuxC::clear();
        }

        /*
         * Apply the preconditioner onto 'in'.
         *
         * \param out: returned vectors
         * \param in: input vectors
         *
         * It applies the deflation on the input vectors and return the results on 'out'.
         *
         *    out = [M^{-1}*Q + K*(I-Q)] * in,
         *
         * where Q = M_oo^{-1}*P*M_oo, P is a projector on M, and K approximates M^{-1}_oo*M_oo.
         */

        std::vector<LinearSolverResults>
        operator()(QPhiXSpinor &out, const QPhiXSpinor &in, ResiduumType resid_type = RELATIVE,
                   InitialGuess guess = InitialGuessNotGiven) const override {
            assert(out.GetNCol() == in.GetNCol());
            IndexType ncol = out.GetNCol();

            std::shared_ptr<QPhiXSpinorF> in_f = AuxQF::tmp(in.GetInfo(), ncol);
            ZeroVec(*in_f);
            ConvertSpinor(in, *in_f, _subset);
            std::shared_ptr<QPhiXSpinorF> out_f = AuxQF::tmp(in.GetInfo(), ncol);
            std::vector<LinearSolverResults> res = operator()(*out_f, *in_f, resid_type, guess);
            ZeroVec(out, _subset.complementary());
            ConvertSpinor(*out_f, out, _subset);
            return res;
        }

        std::vector<LinearSolverResults>
        operator()(QPhiXSpinorF &out, const QPhiXSpinorF &in, ResiduumType resid_type = RELATIVE,
                   InitialGuess guess = InitialGuessNotGiven) const override {
           return apply(out, in, resid_type, guess);
        }

        std::vector<LinearSolverResults>
        operator()(CoarseSpinor &out, const CoarseSpinor &in, ResiduumType resid_type = RELATIVE,
                   InitialGuess guess = InitialGuessNotGiven) const {
           return apply(out, in, resid_type, guess);
        }

        template <typename Spinor>
        std::vector<LinearSolverResults>
        apply(Spinor &out, const Spinor &in, ResiduumType resid_type = RELATIVE,
                   InitialGuess guess = InitialGuessNotGiven) const {
            (void)guess;
            (void)resid_type;

            switch (_style) {
            case 0: // Do K*in
                apply_sub_precon(out, in, do_K);
                break;
            case 1: // Do [A^{-1}*Q + K*(I-Q)]*in
                apply_sub_precon(out, in, do_invM_Q_plus_K_complQ);
                break;
            case 2: // Do [A^{-1}*Q + (I-P)*K*(I-Q)]*in
                apply_sub_precon(out, in, do_invM_Q_plus_complP_K_complQ);
                break;
            default:
                assert(false);
            }
            return std::vector<LinearSolverResults>(in.GetNCol(), LinearSolverResults());
        }

        /**
         * Return M^{-1} * Q * in if _mode==0 else M^{-1}_oo * Q * in
         *
         * \param eo_solver: invertor on _M_fine
         * \param out: (out) output vector
         * \param in: input vector
         */

        template <typename Spinor> void apply_invM_Q(Spinor &out, const Spinor &in) const {
            using AuxS = AuxiliarySpinors<Spinor>;

            assert(in.GetNCol() == out.GetNCol());
            int ncol = in.GetNCol();

            if (_mode < 2 || in.GetInfo().GetLevel() >= _mg_levels->coarse_levels.size()) {
                std::shared_ptr<Spinor> in0 = AuxS::tmp(in);
                ZeroVec(*in0);
                CopyVec(*in0, in, _subset);
                _mg_deflation->VV(out, *in0);
            } else {
                apply_vcycle(out, in);
            }

            ZeroVec(out, _subset.complementary());
        }

        /*
         * Apply K. out = K * in.
         *
         * \param out: returned vectors
         * \param in: input vectors
         */

        void apply_K(QPhiXSpinorF &out, const QPhiXSpinorF &in) const {
            assert(out.GetNCol() == in.GetNCol());
            IndexType ncol = out.GetNCol();

            if (_prec_mode == 0) {
                _M_fine->M_oo_inv(out, in);
            } else {
                std::shared_ptr<CoarseSpinor> in_c = AuxC::tmp(in.GetInfo(), ncol);
                ZeroVec(*in_c, _subset.complementary());
                ConvertSpinor(in, *in_c, _subset);
                std::shared_ptr<CoarseSpinor> out_c = AuxC::tmp(in.GetInfo(), ncol);
                apply_K(*out_c, *in_c);
                ZeroVec(out, _subset.complementary());
                ConvertSpinor(*out_c, out, _subset);
            }
        }

        /*
         * Apply K. out = K * in.
         *
         * \param out: returned vectors
         * \param in: input vectors
         */

        void apply_K(CoarseSpinor &out, const CoarseSpinor &in) const {
            assert(out.GetNCol() == in.GetNCol());
            const unsigned int level = in.GetInfo().GetLevel();

            if (_prec_mode == 0) {
                _mg_levels->coarse_levels[level - 1].M->M_oo_inv(out, in);
            } else if (_K_distance == 0 || !_K_vals[level]) {
                CopyVec(out, in, _subset);

            } else if (_K_distance == 1) {
                // Apply the diagonal of K
#pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    for (int cb = _subset.start; cb < _subset.end; ++cb) {
                        _op[level]->M_diag(out, *_K_vals[level], in, cb, LINOP_OP, tid);
                    }
                }

            } else {
                assert(false);
            }
        }

        const std::shared_ptr<const QPhiXWilsonCloverEOLinearOperatorF> GetM() const {
            return _M_fine;
        }

        const std::shared_ptr<MGDeflation> GetMGDeflation() const { return _mg_deflation; }

        /**
         * Return the lattice information
         */

        const LatticeInfo &GetInfo() const { return _M_fine->GetInfo(); }

        const LatticeInfo &GetInfo(unsigned int level) const {
            if (level == 0) return _M_fine->GetInfo();
            return *_mg_levels->coarse_levels[level - 1].info;
        }

        /**
         * Return the support of the operator (SUBSET_EVEN, SUBSET_ODD, SUBSET_ALL)
         */

        const CBSubset &GetSubset() const { return _subset; }

    private:
        template <typename Spinor> struct S : public ImplicitLinearSolver<Spinor> {
            S(const ALIPrec &aliprec, unsigned int level = 0)
                : ImplicitLinearSolver<Spinor>(aliprec.GetInfo(level), aliprec.GetSubset()),
                  _aliprec(aliprec) {}

            std::vector<LinearSolverResults>
            operator()(Spinor &out, const Spinor &in,
                       ResiduumType resid_type = RELATIVE,
                       InitialGuess guess = InitialGuessNotGiven) const override {
                (void)resid_type;
                (void)guess;
                _aliprec(out, in);
                return std::vector<LinearSolverResults>(in.GetNCol(), LinearSolverResults());
            }

            const ALIPrec &_aliprec;
        };

        /**
         * Return (I-Q) * in
         *
         * \param out: (out) output vector
         * \param in: input vector
         */

        void apply_complQ(QPhiXSpinor &out, const QPhiXSpinor &in) const {
            assert(out.GetNCol() == in.GetNCol());
            IndexType ncol = out.GetNCol();

            std::shared_ptr<QPhiXSpinorF> in_f = AuxQF::tmp(in.GetInfo(), ncol);
            ZeroVec(*in_f);
            ConvertSpinor(in, *in_f, _subset);
            std::shared_ptr<QPhiXSpinorF> out_f = AuxQF::tmp(in.GetInfo(), ncol);
            apply_complQ<QPhiXSpinorF>(*out_f, *in_f);
            ZeroVec(out);
            ConvertSpinor(*out_f, out, _subset);
        }

        /**
         * Return (I-Q) * in
         *
         * \param out: (out) output vector
         * \param in: input vector
         * \param VVin (out): where to save M^{-1}*Q*in
         */

        template <typename Spinor>
        void apply_complQ(Spinor &out, const Spinor &in, Spinor *VVin = nullptr) const {
            using AuxS = AuxiliarySpinors<Spinor>;

            assert(out.GetNCol() == in.GetNCol());
            assert(!VVin || VVin->GetNCol() == in.GetNCol());

            // VVin = M^{-1}*Q*in
            std::shared_ptr<Spinor> VVin0;
            if (!VVin) {
                VVin0 = AuxS::tmp(in);
                VVin = VVin0.get();
            }
            apply_invM_Q(*VVin, in);

            // AVVin = M*VVin
            std::shared_ptr<Spinor> AVVin = AuxS::tmp(in);
            apply_M(*AVVin, *VVin, _mode == 0 ? do_unprec : do_prec);
            VVin0.reset();
            VVin = nullptr;

            // out = in - AVVin
            CopyVec(out, in, _subset);
            YmeqXVec(*AVVin, out, _subset);
            ZeroVec(out, _subset.complementary());
        }

        /**
         * Return (I-P) * in
         *
         * \param out: (out) output vector
         * \param in: input vector
         */

        template <typename Spinor>
        void apply_complP(Spinor &out, const Spinor &in) const {
            using AuxS = AuxiliarySpinors<Spinor>;

            assert(out.GetNCol() == in.GetNCol());

            // A_in = M_in
            std::shared_ptr<Spinor> A_in = AuxS::tmp(in);
            apply_M(*A_in, in, _mode == 0 ? do_unprec : do_prec);

            // VVA_in = M^{-1}*Q*in == P*M^{-1}*in
            std::shared_ptr<Spinor> VVA_in = AuxS::tmp(in);
            apply_invM_Q(*VVA_in, *A_in);
            A_in.reset();

            // out = in - VVA_in
            CopyVec(out, in, _subset);
            YmeqXVec(*VVA_in, out, _subset);
            ZeroVec(out, _subset.complementary());
        }

        enum EOMode {
            do_unprec,  // Multiply by M
            do_prec     // Multiply by (M_oo - M_oe*M_ee^{-1}*M_eo)
        };

        /**
         * Return _M_fine * in if mode == do_unprec else _M_fine.unprecOp * in
         *
         * \param out: (out) output vector
         * \param in: input vector
         * \param mode: either do_unprec or do_prec
         */

        void apply_M(QPhiXSpinor &out, const QPhiXSpinor &in, EOMode mode = do_prec) const {
            assert(out.GetNCol() == in.GetNCol());
            IndexType ncol = out.GetNCol();

            std::shared_ptr<QPhiXSpinorF> in_f = AuxQF::tmp(in.GetInfo(), ncol);
            ZeroVec(*in_f, _subset.complementary());
            ConvertSpinor(in, *in_f, _subset);
            std::shared_ptr<QPhiXSpinorF> out_f = AuxQF::tmp(in.GetInfo(), ncol);
            apply_M(*out_f, *in_f, mode);
            ZeroVec(out, _subset.complementary());
            ConvertSpinor(*out_f, out, _subset);
        }


        /**
         * Return _M_fine * in if mode == do_unprec else _M_fine.unprecOp * in
         *
         * \param out: (out) output vector
         * \param in: input vector
         * \param mode: either do_unprec or do_prec
         */

        void apply_M(QPhiXSpinorF &out, const QPhiXSpinorF &in, EOMode mode = do_prec) const {
            assert(out.GetNCol() == in.GetNCol());
            if (mode == do_prec) {
                (*_M_fine)(out, in);
            } else {
                _M_fine->unprecOp(out, in);
            }
        }

        /**
         * Return _M_fine * in if mode == do_unprec else _M_fine.unprecOp * in
         *
         * \param out: (out) output vector
         * \param in: input vector
         * \param mode: either do_unprec or do_prec
         */

        void apply_M(CoarseSpinor &out, const CoarseSpinor &in, EOMode mode = do_prec) const {
            assert(out.GetNCol() == in.GetNCol());
            if (mode == do_prec) {
                (*_mg_levels->coarse_levels[in.GetInfo().GetLevel() - 1].M)(out, in);
            } else {
                _mg_levels->coarse_levels[in.GetInfo().GetLevel() - 1].M->unprecOp(out, in);
            }
        }

        enum Style {
            do_K,                          // Do K*in
            do_invM_Q_plus_K_complQ,       // Do M^{-1}*Q + K*(I-Q)
            do_invM_Q_plus_complP_K_complQ // Do M^{-1}*Q + (I-P)*K*(I-Q)
        };

        /**
         * Return the result of a preconditioner step controlled by style
         *
         * \param out: (out) output vector
         * \param in: input vector
         * \param style: one of the styles
         *
         * For style == do_K
         *   out = K*in
         *
         * For style == do_invM_Q_plus_K_complQ
         *   out = [M^{-1}*Q + K*(I-Q)]*in
         *
         * For style == do_invM_Q_plus_complP_K_complQ
         *   out = [M^{-1}*Q + (I-P)*K*(I-Q)]*in
         */

        void apply_sub_precon(QPhiXSpinor &out, const QPhiXSpinor &in, Style style) const {
            assert(out.GetNCol() == in.GetNCol());
            IndexType ncol = out.GetNCol();

            std::shared_ptr<QPhiXSpinorF> in_f = AuxQF::tmp(in.GetInfo(), ncol);
            ZeroVec(*in_f);
            ConvertSpinor(in, *in_f, _subset);
            std::shared_ptr<QPhiXSpinorF> out_f = AuxQF::tmp(in.GetInfo(), ncol);
            apply_sub_precon<QPhiXSpinorF>(*out_f, *in_f, style);
            ZeroVec(out);
            ConvertSpinor(*out_f, out, _subset);
        }

        /**
         * Return the result of a preconditioner step controlled by style
         *
         * \param out: (out) output vector
         * \param in: input vector
         * \param style: one of the styles
         *
         * For style == do_K
         *   out = K*in
         *
         * For style == do_invM_Q_plus_K_complQ
         *   out = [M^{-1}*Q + K*(I-Q)]*in
         *
         * For style == do_invM_Q_plus_complP_K_complQ
         *   out = [M^{-1}*Q + (I-P)*K*(I-Q)]*in
         */

        template <typename Spinor>
        void apply_sub_precon(Spinor &out, const Spinor &in, Style style) const {
            using AuxS = AuxiliarySpinors<Spinor>;

            assert(out.GetNCol() == in.GetNCol());

            // Do out = K*in if style == do_K
            if (style == do_K) {
                apply_K(out, in);
                return;
            }

            // I_Q_in = (I-Q)*in and out = M^{-1}*Q*in
            std::shared_ptr<Spinor> I_Q_in = AuxS::tmp(in);
            apply_complQ<Spinor>(*I_Q_in, in, &out);

            // K_I_Q_in = K * I_Q_in
            std::shared_ptr<Spinor> K_I_Q_in = AuxS::tmp(in);
            apply_K(*K_I_Q_in, *I_Q_in);

            if (style == do_invM_Q_plus_K_complQ) {
                YpeqXVec(*K_I_Q_in, out, _subset);
            } else if (style == do_invM_Q_plus_complP_K_complQ) {
                // I_P_K = (I-P) * K_I_Q_in
                std::shared_ptr<Spinor> I_P_K = AuxS::tmp(in);
                apply_complP(*I_P_K, *K_I_Q_in);
                YpeqXVec(*I_P_K, out, _subset);
            } else {
                assert(false);
            }
        }

        /**
         * Return the preconditioner for a particular level
         *
         * \param level: multigrid level
         * \param prec (out): pointer to the preconditioner
         */

        void get_precond(unsigned int level, const LinearSolver<QPhiXSpinorF> *&prec) const {
            assert(level == 0);
            prec = _prec_top.get();
        }

        /**
         * Return the preconditioner for a particular level
         *
         * \param level: multigrid level
         * \param prec (out): pointer to the preconditioner
         */

        void get_precond(unsigned int level, const LinearSolver<CoarseSpinor> *&prec) const {
            assert(level <= _prec_coarse.size());
            prec = _prec_coarse[level - 1].get();
        }

        /**
         * Return M^{-1}_oo * (I-Q) * in
         *
         * \param solver: invertor on M
         * \param out: (out) output vector
         * \param in: input vector
         */

        template <typename Spinor>
        void apply_invM_after_defl(const LinearSolver<Spinor> &eo_solver, Spinor &out,
                                   const Spinor &in) const {
            using AuxS = AuxiliarySpinors<Spinor>;

            assert(in.GetNCol() == out.GetNCol());

            // I_Q_in = (I-Q)*in
            std::shared_ptr<Spinor> I_Q_in = AuxS::tmp(in);
            apply_complQ<Spinor>(*I_Q_in, in);

            // out = M^{-1} * (I-Q) * in
            eo_solver(out, *I_Q_in, RELATIVE);
        }

        template <typename Op>
        void build_K(LinearSolverParamsBase solver_params, unsigned int probing_distance,
                     const Op &M, unsigned int blocking = 32) {
            using Spinor = typename Op::Spinor;
            using AuxS = AuxiliarySpinors<Spinor>;

            if (_K_distance == 0 || _prec_mode == 0) return;
            if (_K_distance > 1) throw std::runtime_error("Not implemented 'K_distance' > 1");

            const LinearSolver<Spinor> *prec;
            get_precond(M.GetInfo().GetLevel(), prec);
            FGMRESGeneric::FGMRESSolverGeneric<Spinor> eo_solver(M, solver_params, prec);

            std::shared_ptr<CoarseGauge> K_vals = std::make_shared<CoarseGauge>(M.GetInfo());
            ZeroGauge(*K_vals);

            std::shared_ptr<Coloring> coloring =
                get_good_coloring(M, eo_solver, probing_distance, solver_params.RsdTarget * 2);

            unsigned int num_probing_vecs = coloring->GetNumSpinColorColors();
            for (unsigned int col = 0, nc = std::min(num_probing_vecs, blocking);
                 col < num_probing_vecs;
                 col += nc, nc = std::min(num_probing_vecs - col, blocking)) {

                // p(i) is the probing vector for the color col+i
                std::shared_ptr<Spinor> p = AuxS::tmp(M.GetInfo(), nc);
                coloring->GetProbingVectors(*p, col);

                // sol = inv(M_fine) * (I-P) * p
                std::shared_ptr<Spinor> sol = AuxS::tmp(M.GetInfo(), nc);
                apply_invM_after_defl(eo_solver, *sol, *p);
                p.reset();

                // Update K from sol
                update_K_from_probing_vecs(*K_vals, *coloring, col, sol);
            }

            _K_vals[M.GetInfo().GetLevel()] = K_vals;

            test_K(eo_solver);
        }

        template <typename Spinor>
        void update_K_from_probing_vecs(CoarseGauge &K_vals, const Coloring &coloring,
                                        unsigned int c0, const std::shared_ptr<Spinor> sol) {
            const LatticeInfo info = sol->GetInfo();
            IndexType num_cbsites = info.GetNumCBSites();
            IndexType num_color = info.GetNumColors();
            IndexType num_spin = info.GetNumSpins();
            IndexType ncol = sol->GetNCol();

            // Loop over the sites and sum up the norm
#pragma omp parallel for collapse(3) schedule(static)
            for (int cb = _subset.start; cb < _subset.end; ++cb) {
                for (int cbsite = 0; cbsite < num_cbsites; ++cbsite) {
                    for (int col = 0; col < ncol; ++col) {
                        // Decompose the color into the node's color and the spin-color components
                        IndexType col_spin, col_color, node_color;
                        coloring.SpinColorColorComponents(c0 + col, node_color, col_spin,
                                                          col_color);
                        unsigned int colorj = coloring.GetColorCBIndex(cb, cbsite);

                        // Process this site if its color is the same as the color of the probing
                        // vector
                        if (colorj != (unsigned int)node_color) continue;

                        // Get diag
                        for (int color = 0; color < num_color; ++color) {
                            for (int spin = 0; spin < num_spin; ++spin) {
                                int g5 = (spin < num_spin / 2 ? 1 : -1) *
                                         (col_spin < num_spin / 2 ? 1 : -1);
                                K_vals.GetSiteDiagData(cb, cbsite, col_spin, col_color, spin, color,
                                                       RE) +=
                                    (*sol)(col, cb, cbsite, spin, color, 0) / 2;
                                K_vals.GetSiteDiagData(cb, cbsite, col_spin, col_color, spin, color,
                                                       IM) +=
                                    (*sol)(col, cb, cbsite, spin, color, 1) / 2;
                                K_vals.GetSiteDiagData(cb, cbsite, spin, color, col_spin, col_color,
                                                       RE) +=
                                    (*sol)(col, cb, cbsite, spin, color, 0) / 2 * g5;
                                K_vals.GetSiteDiagData(cb, cbsite, spin, color, col_spin, col_color,
                                                       IM) -=
                                    (*sol)(col, cb, cbsite, spin, color, 1) / 2 * g5;
                            }
                        }
                    }
                }
            }
        }

        template <typename Op, typename Solver>
        std::shared_ptr<Coloring> get_good_coloring(const Op &M, const Solver &eo_solver,
                                                    unsigned int max_probing_distance, double tol) {
            using Spinor = typename Solver::Spinor;
            using AuxS = AuxiliarySpinors<Spinor>;

            const LatticeInfo &info = eo_solver.GetInfo();

            // Returned coloring
            std::shared_ptr<Coloring> coloring;

            // Build probing vectors to get the exact first columns for sites [1 0 0 0]
            IndexArray site = {{1, 0, 0, 0}}; // This is an ODD site
            std::shared_ptr<Spinor> e = AuxS::tmp(info, info.GetNumColorSpins());
            ZeroVec(*e);
            std::vector<IndexType> the_cbsite = Coloring::GetKDistNeighbors(site, 0, info);
            for (unsigned int i = 0; i < the_cbsite.size(); ++i) {
                for (int color = 0; color < info.GetNumColors(); ++color) {
                    for (int spin = 0; spin < info.GetNumSpins(); ++spin) {
                        int sc = color * info.GetNumSpins() + spin;
                        (*e)(sc, ODD, the_cbsite[i], spin, color, 0) = 1.0;
                    }
                }
            }
            std::vector<IndexType> cbsites_dist_k =
                Coloring::GetKDistNeighbors(site, (_K_distance - 1) * 2, info);

            // sol_e = inv(M_fine) * (I-Q) * e
            std::shared_ptr<Spinor> sol_e = AuxS::tmp(info, info.GetNumColorSpins());
            apply_invM_after_defl(eo_solver, *sol_e, *e);
            ZeroVec(*sol_e, _subset.complementary());

            unsigned int probing_distance = 1;
            while (probing_distance <= max_probing_distance) {
                // Create coloring
                coloring = std::make_shared<Coloring>(std::make_shared<LatticeInfo>(info),
                                                      probing_distance, SUBSET_ODD);

                // Get the probing vectors for "site" 
                double color_node = 0;
                if (the_cbsite.size() > 0)
                    color_node = coloring->GetColorCBIndex(ODD, the_cbsite[0]);
                GlobalComm::GlobalSum(color_node);
                std::shared_ptr<Spinor> p = AuxS::tmp(info, info.GetNumColorSpins());
                coloring->GetProbingVectors(
                    *p, coloring->GetSpinColorColor((unsigned int)color_node, 0, 0));

                // sol_p = inv(M_fine) * (I-P) * p
                std::shared_ptr<Spinor> sol_p = AuxS::tmp(info, info.GetNumColorSpins());
                apply_invM_after_defl(eo_solver, *sol_p, *p);

                // Compute sol_F = \sum |sol_e[i,j]|^2 over the K nonzero pattern.
                // Compute diff_F = \sum |sol_e[i,j]-sol_p[i,j]|^2 over the K nonzero pattern.
                double sol_F = 0.0, diff_F = 0.0;
                vector3d<std::complex<float>> sol_p_00(
                    cbsites_dist_k.size(), info.GetNumColorSpins(), info.GetNumColorSpins());
                for (unsigned int i = 0; i < cbsites_dist_k.size(); ++i) {
                    for (int colorj = 0; colorj < info.GetNumColors(); ++colorj) {
                        for (int spinj = 0; spinj < info.GetNumSpins(); ++spinj) {
                            int sc_e_j = colorj * info.GetNumSpins() + spinj;
                            int sc_p_j = coloring->GetSpinColorColor(0, spinj, colorj);
                            for (int color = 0; color < info.GetNumColors(); ++color) {
                                for (int spin = 0; spin < info.GetNumSpins(); ++spin) {
                                    std::complex<float> sol_e_ij(
                                        (*sol_e)(sc_e_j, ODD, i, spin, color, 0),
                                        (*sol_e)(sc_e_j, ODD, i, spin, color, 1));
                                    std::complex<float> sol_p_ij(
                                        (*sol_p)(sc_p_j, ODD, i, spin, color, 0),
                                        (*sol_p)(sc_p_j, ODD, i, spin, color, 1));
                                    sol_F += (sol_e_ij * std::conj(sol_e_ij)).real();
                                    std::complex<float> diff_ij = sol_e_ij - sol_p_ij;
                                    diff_F += (diff_ij * std::conj(diff_ij)).real();

                                    int sc_p_i = coloring->GetSpinColorColor(0, spin, color);
                                    sol_p_00(i, sc_p_j, sc_p_i) = sol_p_ij;
                                }
                            }
                        }
                    }
                }

                GlobalComm::GlobalSum(diff_F);
                GlobalComm::GlobalSum(sol_F);

                // Zero sol_p[i,j] that are not on the K nonzero pattern
                ZeroVec(*sol_p, SUBSET_ALL);
                for (unsigned int i = 0; i < cbsites_dist_k.size(); ++i) {
                    for (int colorj = 0; colorj < info.GetNumColors(); ++colorj) {
                        for (int spinj = 0; spinj < info.GetNumSpins(); ++spinj) {
                            int sc_p_j = coloring->GetSpinColorColor(0, spinj, colorj);
                            for (int color = 0; color < info.GetNumColors(); ++color) {
                                for (int spin = 0; spin < info.GetNumSpins(); ++spin) {
                                    int sc_p_i = coloring->GetSpinColorColor(0, spin, color);
                                    (*sol_p)(sc_p_j, ODD, i, spin, color, 0) =
                                        sol_p_00(i, sc_p_j, sc_p_i).real();
                                    (*sol_p)(sc_p_j, ODD, i, spin, color, 1) =
                                        sol_p_00(i, sc_p_j, sc_p_i).imag();
                                }
                            }
                        }
                    }
                }

                // Compute norm_sol_e = |sol_e|_F
                std::shared_ptr<Spinor> aux = AuxS::tmp(info, info.GetNumColorSpins());
                std::shared_ptr<Spinor> aux0 = AuxS::tmp(info, info.GetNumColorSpins());
                double norm_sol_e = sqrt(sum(Norm2Vec(*sol_e, SUBSET_ODD)));

                // Compute norm_diff = |sol_p-sol_e|_F
                CopyVec(*aux, *sol_e);
                double norm_diff = sqrt(sum(XmyNorm2Vec(*aux, *sol_p, SUBSET_ODD)));

                // Compute norm_F = |A*(sol_p - sol_e)|_F
                M(*aux0, *aux);
                double norm_F = sqrt(sum(Norm2Vec(*aux0, SUBSET_ODD)));

                MasterLog(INFO,
                          "K probing error with %d distance coloring: %d colors               "
                          "||M^{-1}_00-K_00||_F/||M^{-1}_00||_F= "
                          "%g ||M^{-1}_0-K_0||_F/||M^{-1}_0||_F= %g   ||M*K-I||= %g",
                          probing_distance, (int)coloring->GetNumSpinColorColors(),
                          sqrt(diff_F / sol_F), norm_diff / norm_sol_e, norm_F);

                // Compute norm_diff = |sol_p-sol_e|_F
                CopyVec(*aux, *sol_p);
                apply_complP(*sol_p, *aux);

                CopyVec(*aux, *sol_e);
                norm_diff = sqrt(sum(XmyNorm2Vec(*aux, *sol_p, SUBSET_ODD)));

                // Compute norm_F = |A*(sol_p - sol_e)|_F
                M(*aux0, *aux);
                norm_F = sqrt(sum(Norm2Vec(*aux0, SUBSET_ODD)));

                MasterLog(INFO,
                          "K probing error with %d distance coloring: %d colors [with postdef]"
                          "||M^{-1}_00-K_00||_F/||M^{-1}_00||_F= "
                          "%g ||M^{-1}_0-K_0||_F/||M^{-1}_0||_F= %g   ||M*K-I||= %g",
                          probing_distance, (int)coloring->GetNumSpinColorColors(),
                          sqrt(diff_F / sol_F), norm_diff / norm_sol_e, norm_F);

                if (diff_F <= sol_F * tol * tol) break;

                probing_distance++;
                // Coloring produces distinct coloring schemes for even distances only (excepting
                // 1-distance)
                if (probing_distance % 2 == 1) probing_distance++;
            }

            return coloring;
        }

        template <typename Spinor>
        void test_K(const LinearSolver<Spinor> &eo_solver) {
            using AuxS = AuxiliarySpinors<Spinor>;
            const LatticeInfo &info = eo_solver.GetInfo();

            // Build probing vectors to get the exact first columns for ODD site 0
            const int nc = info.GetNumColorSpins();
            std::shared_ptr<Spinor> e = AuxS::tmp(info, nc);
            ZeroVec(*e);
            if (info.GetNodeInfo().NodeID() == 0) {
                for (int color = 0; color < info.GetNumColors(); ++color) {
                    for (int spin = 0; spin < info.GetNumSpins(); ++spin) {
                        int sc = color * info.GetNumSpins() + spin;
                        (*e)(sc, ODD, 0, spin, color, 0) = 1.0;
                    }
                }
            }

            // I_Q_e = (I-Q)*e
            std::shared_ptr<Spinor> I_Q_e = AuxS::tmp(info, nc);
            apply_complQ<Spinor>(*I_Q_e, *e);
            e.reset();

            // sol_e = inv(M) * (I-Q) * e
            std::shared_ptr<Spinor> sol_e = AuxS::tmp(info, nc);
            eo_solver(*sol_e, *I_Q_e);

            // sol_p \approx K * (I-Q) * e
            std::shared_ptr<Spinor> sol_p = AuxS::tmp(info, nc);
            apply_K(*sol_p, *I_Q_e);

            double norm_e = sqrt(sum(Norm2Vec(*sol_e, SUBSET_ODD)));
            double norm_diff = sqrt(sum(XmyNorm2Vec(*sol_e, *sol_p, SUBSET_ODD)));
            MasterLog(INFO, "K probing error : ||M^{-1}-K||_F/||M^{-1}||_F= %g",
                      norm_diff / norm_e);
        }

        void apply_vcycle(QPhiXSpinorF &out, const QPhiXSpinorF &in) const {
            assert(out.GetNCol() == in.GetNCol());
            IndexType ncol = out.GetNCol();

            // Quick exit
            if (_mg_levels->coarse_levels.size() <= 0) {
                CopyVec(out, in, _subset);
                return;
            }

            std::shared_ptr<CoarseSpinor> coarse_in =
                AuxC::tmp(*_mg_levels->coarse_levels[0].info, ncol);
            std::shared_ptr<CoarseSpinor> coarse_out =
                AuxC::tmp(*_mg_levels->coarse_levels[0].info, ncol);
            _Transfer_fine_level->R(in, ODD, *coarse_in);
            (*_bottom_solver[0])(*coarse_out, *coarse_in);
            _Transfer_fine_level->P(*coarse_out, ODD, out);
        }

        void apply_vcycle(CoarseSpinor &out, const CoarseSpinor &in) const {
            assert(out.GetNCol() == in.GetNCol());
            IndexType ncol = out.GetNCol();
            const unsigned int level = in.GetInfo().GetLevel();

            // Quick exit
            if (_mg_levels->coarse_levels.size() <= level) {
                CopyVec(out, in, _subset);
                return;
            }

            std::shared_ptr<CoarseSpinor> coarse_in =
                AuxC::tmp(*_mg_levels->coarse_levels[level].info, ncol);
            std::shared_ptr<CoarseSpinor> coarse_out =
                AuxC::tmp(*_mg_levels->coarse_levels[level].info, ncol);
            _Transfer_coarse_level[level - 1]->R(in, ODD, *coarse_in);
            (*_bottom_solver[level])(*coarse_out, *coarse_in);
            _Transfer_coarse_level[level - 1]->P(*coarse_out, ODD, out);
        }

        const std::shared_ptr<const QPhiXWilsonCloverEOLinearOperatorF> _M_fine;
        std::shared_ptr<MGDeflation> _mg_deflation;
        std::vector<std::shared_ptr<CoarseGauge>> _K_vals;
        unsigned int _K_distance;
        std::vector<std::shared_ptr<const CoarseDiracOp>> _op;
        const CBSubset _subset;
        const unsigned int _mode;
        const unsigned int _style;
        const unsigned int _prec_mode;
        std::shared_ptr<QPhiXMultigridLevelsEO> _mg_levels;
        std::shared_ptr<const S<QPhiXSpinorF>> _prec_top;
        std::vector<std::shared_ptr<const S<CoarseSpinor>>> _prec_coarse;
        std::shared_ptr<QPhiXTransfer<QPhiXSpinorF>> _Transfer_fine_level;
        std::vector<std::shared_ptr<CoarseTransfer>> _Transfer_coarse_level;
        std::vector<std::shared_ptr<const LinearSolver<CoarseSpinor>>> _bottom_solver;
    };
} // namespace MG

#endif // INCLUDE_LATTICE_QPHIX_QPHIX_ALI_H_
