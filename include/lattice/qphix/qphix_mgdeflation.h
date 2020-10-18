/*
 * qphix_mgdeflation.h
 *
 *  Created on: May 29, 2020
 *      Author: Eloy Romero <eloy@cs.wm.edu>
 */

#ifndef INCLUDE_LATTICE_QPHIX_QPHIX_DISCO_H_
#define INCLUDE_LATTICE_QPHIX_QPHIX_DISCO_H_

#include "lattice/coarse/coarse_deflation.h"               // computeDeflation
#include "lattice/coarse/coarse_transfer.h"                // CoarseTransfer
#include "lattice/coarse/invfgmres_coarse.h"               // UnprecFGMRESSolverCoarseWrapper
#include "lattice/eigs_common.h"                           // EigsParams
#include "lattice/fine_qdpxx/mg_params_qdpxx.h"            // SetupParams
#include "lattice/qphix/mg_level_qphix.h"                  // QPhiXMultigridLevels
#include "lattice/qphix/qphix_eo_clover_linear_operator.h" // QPhiXWilsonCloverEOLinearOperatorF
#include "lattice/qphix/qphix_transfer.h"                  // QPhiXTransfer
#include "lattice/qphix/qphix_types.h"                     // QPhiXSpinorF
#include <MG_config.h>
#include <cassert>
#include <memory>
#include <stdexcept>

namespace MG {

    /*
	 * Deflate an approximate invariant subspace of the multigrid prolongator
	 *
	 * If V is a subset of the multigrid prolongator P at the coarsest level on a
	 * linear operator A, then MGDeflation::operator()(out, in) does:
	 *
	 *   out = \gamma_5 * V * inv(V^H * A * \gamma_5 * V) * V^H * A * in,
	 *
	 * where V^H * A * \gamma_5 * V is numerically close to a diagonal matrix.
	 * This is achieved by computing the right singular vectors of inv(P^H * A * P), X, with
	 * MG::computeDeflation and setting V as P * X.
	 */

    class MGDeflation : public AuxiliarySpinors<QPhiXSpinorF>,
                        public AuxiliarySpinors<CoarseSpinor> {
        using AuxQ = AuxiliarySpinors<QPhiXSpinorF>;
        using AuxC = AuxiliarySpinors<CoarseSpinor>;

    public:
        /*
         * Constructor
         *
         * \param M_fine: original operator
         * \param p: Multigrid parameters
         * \param solver_params: eigensolver's invertor operator parameters
         * \param eigs_param: eigensolver parameters
         */

        MGDeflation(const std::shared_ptr<const QPhiXWilsonCloverEOLinearOperatorF> M_fine,
                    SetupParams p, LinearSolverParamsBase solver_params, EigsParams eigs_params)
            : _M_fine(M_fine) {

            // Setup multigrid
            p.purpose = SetupParams::INVARIANT_SPACE;
            _mg_levels = std::make_shared<QPhiXMultigridLevelsEO>();
            SetupQPhiXMGLevels(p, *_mg_levels, _M_fine);

            setup(solver_params, eigs_params);
        }

        /*
         * Constructor
         *
         * \param M_fine: original operator
         * \param mg_levels: Multigrid structure
         * \param solver_params: eigensolver's invertor operator parameters
         * \param eigs_param: eigensolver parameters
         */

        MGDeflation(const std::shared_ptr<const QPhiXWilsonCloverEOLinearOperatorF> M_fine,
                    std::shared_ptr<QPhiXMultigridLevelsEO> mg_levels,
                    LinearSolverParamsBase solver_params, EigsParams eigs_params)
            : _M_fine(M_fine), _mg_levels(mg_levels) {
            setup(solver_params, eigs_params);
        }

        /*
         * Setup multigrid transfer and compute deflation space
         *
         * \param solver_params: eigensolver's invertor operator parameters
         * \param eigs_param: eigensolver parameters
         */

        void setup(LinearSolverParamsBase solver_params, EigsParams eigs_params) {
            AuxQ::subrogateTo(_M_fine.get());

            if (_mg_levels->coarse_levels.size() <= 0) return;

            // Generate the prolongators and restrictiors
            _Transfer_coarse_level.resize(_mg_levels->coarse_levels.size() - 1);
            for (int coarse_idx = (int)_mg_levels->coarse_levels.size() - 2; coarse_idx >= 0;
                 --coarse_idx) {
                _Transfer_coarse_level[coarse_idx] = std::make_shared<CoarseTransfer>(
                    _mg_levels->coarse_levels[coarse_idx].blocklist,
                    _mg_levels->coarse_levels[coarse_idx].null_vecs);
            }
            _Transfer_fine_level = std::make_shared<QPhiXTransfer<QPhiXSpinorF>>(
                _mg_levels->fine_level.blocklist, _mg_levels->fine_level.null_vecs);

            // Create the solver for the coarsest level
            auto this_level_linop = _mg_levels->coarse_levels.back().M;
            solver_params.RsdTarget =
                std::min(solver_params.RsdTarget, eigs_params.RsdTarget * 0.6);
            UnprecFGMRESSolverCoarseWrapper solver(*this_level_linop, solver_params, nullptr);
            test_coarse_solver(solver);

            // Compute the largest right singular vectors of inv(P^H * A * P), that is,
            // inv(P^H * A * P)*_eigenvectors[i] == \gamma_5 * _eigenvector[i] * _eigenvalue[i]
            MasterLog(INFO, "Computing deflation for level %d rank= %d", (int)_mg_levels->coarse_levels.size(), eigs_params.MaxNumEvals);
            computeDeflation(*_mg_levels->coarse_levels.back().info, solver, eigs_params,
                             _eigenvectors, _eigenvalues);

            // Check eigendecomposition: (P^H * A *P) * \gamma_5 * _eigenvector[i] * _eigenvalue[i] == _eigenvector[i]
            if (0) {
                const LatticeInfo &Minfo = *_mg_levels->coarse_levels.back().info;
                CoarseSpinor g5eigenvector(Minfo, _eigenvectors->GetNCol()),
                             Mg5eigenvector_lambda(Minfo, _eigenvectors->GetNCol());
                CopyVec(g5eigenvector, *_eigenvectors);
                Gamma5Vec(g5eigenvector);
                ZeroVec(Mg5eigenvector_lambda);
                this_level_linop->unprecOp(Mg5eigenvector_lambda, g5eigenvector);
                ScaleVec(_eigenvalues, Mg5eigenvector_lambda);
                std::vector<double> rnorms2 = XmyNorm2Vec(Mg5eigenvector_lambda, *_eigenvectors);
                for (unsigned int i = 0; i < rnorms2.size(); ++i)
                    MasterLog(INFO, "Eigenpair error %d  %g", i, sqrt(rnorms2[i]));
            }

            AuxQ::clear();
            AuxC::clear();
        }

        /**
         * Form of the action
         */

        enum Form {
            do_AVV,     // Do AV(V'AV)^{-1}V'
            do_VVA,     // Do V(V'AV)^{-1}AV'
            do_VV       // Do V(V'AV)^{-1}V
        };
    
        /*
         * Apply oblique deflation on vectors 'in'.
         *
         * \param out: returned vectors
         * \param in: input vectors
         *
         * It applies the deflation on the input vectors and return the results on 'out':
         *
         *   out = \gamma_5 * V * inv(V^H * A * \gamma_5 * V) * V^H * A * in,
         */

        template <class Spinor> void VVA(Spinor &out, const Spinor &in) const {
            apply(out, in, do_VVA);
        }

        /*
         * Apply oblique deflation on vectors 'in'.
         *
         * \param out: returned vectors
         * \param in: input vectors
         *
         * It applies the deflation on the input vectors and return the results on 'out':
         *
         *   out = A * \gamma_5 * V * inv(V^H * A * \gamma_5 * V) * V^H * in,
         */

        template <class Spinor> void AVV(Spinor &out, const Spinor &in) const {
            apply(out, in, do_AVV);
        }

        /*
         * Apply oblique deflation on vectors 'in'.
         *
         * \param out: returned vectors
         * \param in: input vectors
         *
         * It applies the deflation on the input vectors and return the results on 'out':
         *
         *   out = \gamma_5 * V * inv(V^H * A * \gamma_5 * V) * V^H * A * in,
         */

        template <class Spinor> void operator()(Spinor &out, const Spinor &in) const {
            VVA(out, in, do_VVA);
        }

        /*
         * Apply the A^{-1} * P * in or P * A^{-1} * in.
         *
         * \param out: returned vectors
         * \param in: input vectors
         *
         * Return the results on 'out':
         *
         *   out = \gamma_5 * V * inv(V^H * A * \gamma_5 * V) * V^H * in,
         */

        template <class Spinor> void VV(Spinor &out, const Spinor &in) const {
            apply(out, in, do_VV);
        }

        /**
         * Apply oblique deflation on vectors 'in'.
         *
         * \param out: returned vectors
         * \param in: input vectors
         * \param form: if do_VVA,
         *          out = \gamma_5 * V * inv(V^H * A * \gamma_5 * V) * V^H * A * in,
         *        elseif do_AVV
         *          out = A * \gamma_5 * V * inv(V^H * A * \gamma_5 * V) * V^H * in.
         *        elseif do_VV
         *          out = \gamma_5 * V * inv(V^H * A * \gamma_5 * V) * V^H * in.
         *
         * It applies the deflation on the input vectors and return the results on 'out'.
         */

        void apply(QPhiXSpinor &out, const QPhiXSpinor &in, Form form) const {
            AssertCompatible(in.GetInfo(), out.GetInfo());
            assert(out.GetNCol() == in.GetNCol());
            IndexType ncol = out.GetNCol();

            // Convert to float
            std::shared_ptr<QPhiXSpinorF> in_f = AuxQ::tmp(*_mg_levels->fine_level.info, ncol);
            ZeroVec(*in_f, SUBSET_ALL);
            ConvertSpinor(in, *in_f);

            // apply projector
            std::shared_ptr<QPhiXSpinorF> out_f = AuxQ::tmp(*_mg_levels->fine_level.info, ncol);
            apply(*out_f, *in_f, form);

            // Convert back to double
            ConvertSpinor(*out_f, out);
        }

        /**
         * Apply oblique deflation on vectors 'in'.
         *
         * \param out: returned vectors
         * \param in: input vectors
         * \param form: if do_VVA,
         *          out = \gamma_5 * V * inv(V^H * A * \gamma_5 * V) * V^H * A * in,
         *        elseif do_AVV
         *          out = A * \gamma_5 * V * inv(V^H * A * \gamma_5 * V) * V^H * in.
         *        elseif do_VV
         *          out = \gamma_5 * V * inv(V^H * A * \gamma_5 * V) * V^H * in.
         *
         * It applies the deflation on the input vectors and return the results on 'out'.
         */

         void apply(QPhiXSpinorF &out, const QPhiXSpinorF &in_f, Form form) const {
            AssertCompatible(in_f.GetInfo(), out.GetInfo());
            assert(out.GetNCol() == in_f.GetNCol());
            IndexType ncol = out.GetNCol();

            // Ain = A * in if do_VVA else in
            std::shared_ptr<QPhiXSpinorF> Ain_f = AuxQ::tmp(*_mg_levels->fine_level.info, ncol);
            if (form == do_VVA) {
                Ain_f = AuxQ::tmp(*_mg_levels->fine_level.info, ncol);
                ZeroVec(*Ain_f, SUBSET_ALL);
                _M_fine->unprecOp(*Ain_f, in_f, LINOP_OP);
            } else {
                CopyVec(*Ain_f, in_f);
            }

            // Transfer to first coarse level
            std::shared_ptr<CoarseSpinor> coarse_in =
                AuxC::tmp(*_mg_levels->coarse_levels[0].info, ncol);
            _Transfer_fine_level->R(*Ain_f, *coarse_in);
            Ain_f.reset();

            // Compute course_out = \gamma_5 * inv(V^H * A * \gamma_5 * V) * coarse_in
            std::shared_ptr<CoarseSpinor> coarse_out =
                AuxC::tmp(*_mg_levels->coarse_levels[0].info, ncol);
            apply(*coarse_out, *coarse_in, do_VV);

            // Transfer to the fine level
            std::shared_ptr<QPhiXSpinorF> out_f = AuxQ::tmp(*_mg_levels->fine_level.info, ncol);
            _Transfer_fine_level->P(*coarse_out, *out_f);

            // out = out_f if do_VVA else A*out_f
            if (form == do_AVV) {
                ZeroVec(out, SUBSET_ALL);
                _M_fine->unprecOp(out, *out_f, LINOP_OP);
            } else {
                CopyVec(out, *out_f);
            }
        }

        /**
         * Apply oblique deflation on vectors 'in'.
         *
         * \param out: returned vectors
         * \param in: input vectors
         * \param form: if do_VVA,
         *          out = \gamma_5 * V * inv(V^H * A * \gamma_5 * V) * V^H * A * in,
         *        elseif do_AVV
         *          out = A * \gamma_5 * V * inv(V^H * A * \gamma_5 * V) * V^H * in.
         *        elseif do_VV
         *          out = \gamma_5 * V * inv(V^H * A * \gamma_5 * V) * V^H * in.
         *
         * It applies the deflation on the input vectors and return the results on 'out'.
         */

        void apply(CoarseSpinor &out, const CoarseSpinor &in, Form form) const {
            AssertCompatible(in.GetInfo(), out.GetInfo());
            assert(out.GetNCol() == in.GetNCol());
            IndexType ncol = out.GetNCol();
            const unsigned int coarse_level = in.GetInfo().GetLevel();
            assert(coarse_level >= 1 && coarse_level - 1 < _mg_levels->coarse_levels.size());

            // Ain = A * in if do_VVA else in
            std::shared_ptr<CoarseSpinor> coarse_in =
                AuxC::tmp(*_mg_levels->coarse_levels[coarse_level - 1].info, ncol);
            if (form == do_VVA) {
                _mg_levels->coarse_levels[coarse_level - 1].M->unprecOp(*coarse_in, in, LINOP_OP);
            } else {
                CopyVec(*coarse_in, in);
            }

            // Transfer to the deepest level
            for (int coarse_idx = coarse_level + 1; // target level
                 coarse_idx < (int)_mg_levels->coarse_levels.size() + 1; coarse_idx++) {
                assert(coarse_idx >= 1 && coarse_idx - 1 < _mg_levels->coarse_levels.size());
                std::shared_ptr<CoarseSpinor> in_level =
                    AuxC::tmp(*_mg_levels->coarse_levels[coarse_idx - 1].info, ncol);
                assert(coarse_idx >= 2 && coarse_idx - 2 < _Transfer_coarse_level.size());
                _Transfer_coarse_level[coarse_idx - 2]->R(*coarse_in, *in_level);
                coarse_in = in_level;
            }

            // Compute course_out = \gamma_5 * inv(V^H * A * \gamma_5 * V) * coarse_in
            std::shared_ptr<CoarseSpinor> coarse_out =
                AuxC::tmp(*_mg_levels->coarse_levels.back().info, ncol);
            project_coarse_level(*coarse_out, *coarse_in);
            coarse_in.reset();

            // Transfer to the first coarse level
            for (int coarse_idx = (int)_mg_levels->coarse_levels.size() - 1; // target level
                 coarse_idx >= (int)coarse_level; coarse_idx--) {
                assert(coarse_idx >= 1 && coarse_idx - 1 < _mg_levels->coarse_levels.size());
                std::shared_ptr<CoarseSpinor> out_level =
                    AuxC::tmp(*_mg_levels->coarse_levels[coarse_idx - 1].info, ncol);
                assert(coarse_idx >= 1 && coarse_idx - 1 < _Transfer_coarse_level.size());
                _Transfer_coarse_level[coarse_idx - 1]->P(*coarse_out, *out_level);
                coarse_out = out_level;
            }

            // out = A*coarse_out if do_AVV else coarse_out
            if (form == do_AVV) {
                _mg_levels->coarse_levels[coarse_level - 1].M->unprecOp(out, *coarse_out, LINOP_OP);
            } else {
                CopyVec(out, *coarse_out);
            }
        }

        /*
         * Return the columns of V starting from i-th column.
         *
         * \param i: index of the first column to return
         * \param out: output vectors
         */

        template <class Spinor> void V(unsigned int i, Spinor &out) const {
            assert(out.GetNCol() + i <= _eigenvectors->GetNCol());
            IndexType ncol = out.GetNCol();

            // Apply deflation at the coarsest level
            std::shared_ptr<CoarseSpinor> coarse_out =
                AuxC::tmp(*_mg_levels->coarse_levels.back().info, ncol);
            CopyVec(*coarse_out, 0, ncol, *_eigenvectors, i);

            // Transfer to the first coarse level
            for (int coarse_idx = (int)_mg_levels->coarse_levels.size() - 2; coarse_idx >= 0;
                 coarse_idx--) {
                std::shared_ptr<CoarseSpinor> out_level =
                    AuxC::tmp(*_mg_levels->coarse_levels[coarse_idx].info, ncol);
                _Transfer_coarse_level[coarse_idx]->P(*coarse_out, *out_level);
                coarse_out = out_level;
            }

            // Transfer to the fine level
            std::shared_ptr<QPhiXSpinorF> out_f = AuxQ::tmp(*_mg_levels->fine_level.info, ncol);
            _Transfer_fine_level->P(*coarse_out, *out_f);

            // Convert back to double
            ConvertSpinor(*out_f, out);
        }

        /*
         * Return the columns of \gamma_5 * V starting from i-th column.
         *
         * \param i: index of the first column to return
         * \param out: output vectors
         */

        template <class Spinor> void g5V(unsigned int i, Spinor &out) const {
            assert(out.GetNCol() + i <= _eigenvectors->GetNCol());
            IndexType ncol = out.GetNCol();

            // Apply deflation at the coarsest level
            std::shared_ptr<CoarseSpinor> coarse_out =
                AuxC::tmp(*_mg_levels->coarse_levels.back().info, ncol);
            CopyVec(*coarse_out, 0, ncol, *_eigenvectors, i);
            Gamma5Vec(*coarse_out);

            // Transfer to the first coarse level
            for (int coarse_idx = (int)_mg_levels->coarse_levels.size() - 2; coarse_idx >= 0;
                 coarse_idx--) {
                std::shared_ptr<CoarseSpinor> out_level =
                    AuxC::tmp(*_mg_levels->coarse_levels[coarse_idx].info, ncol);
                _Transfer_coarse_level[coarse_idx]->P(*coarse_out, *out_level);
                coarse_out = out_level;
            }

            // Transfer to the fine level
            std::shared_ptr<QPhiXSpinorF> out_f = AuxQ::tmp(*_mg_levels->fine_level.info, ncol);
            _Transfer_fine_level->P(*coarse_out, *out_f);

            // Convert back to double
            ConvertSpinor(*out_f, out);
        }

        /*
         * Return the diagonal of inv(V^H * A \* V)
         */

        std::vector<std::complex<double>> GetDiagInvCoarse() const {

            std::vector<std::complex<double>> d(_eigenvalues.size());
            for (unsigned int i = 0; i < d.size(); i++) d[i] = _eigenvalues[i];

            return d;
        }

        const LatticeInfo &GetInfo() { return _M_fine->GetInfo(); }

        const std::shared_ptr<const QPhiXWilsonCloverEOLinearOperatorF> GetM() const {
            return _M_fine;
        }

        unsigned int GetRank() { return _eigenvalues.size(); }

    private:
        /**
         * Solver on the restricted coarsest operator, \gamma_5 * X * \Lambda * X^H * in
         *
         * \param out: returned vectors
         * \param in: input vectors
         *
         * Return out = \gamma_5 * X * \Lambda * X^H * in, where \lambda[i] and X[i] are the
         * _eigenvalues[i] and _eigenvectors[i] of \gamma_5 * inv(P^H * A * P), where P is
         * the coarsest restrictor. If \lambda[i] and X[i] are the exact eigenpairs then,
         *
         *  out = \gamma_5 * X * \Lambda * X^H * in
         *      = \gamma_5 * X * inv(X^H * P^H * A * P * \gamma_5 * X) * X^H * in
         *      = \gamma_5 * X * inv(V^H * A * \gamma_5 * V)*X^H
         */

        void project_coarse_level(CoarseSpinor &out, const CoarseSpinor &in) const {
            assert(out.GetNCol() == in.GetNCol());
            unsigned int ncol = (unsigned int)out.GetNCol();

            // Do ip = X' * in
            unsigned int nEv = _eigenvectors->GetNCol();
            std::vector<std::complex<double>> ip = InnerProductMat(*_eigenvectors, in);

            // Do ip = \Lambda * ip
            for (unsigned int j = 0; j < ncol; ++j)
                for (unsigned int i = 0; i < nEv; ++i) ip[i + j * nEv] *= _eigenvalues[i];

            // out = X * ip
            UpdateVecs(*_eigenvectors, ip, out);

            // Apply gamma_5
            Gamma5Vec(out);
        }

        template <typename Solver> void test_coarse_solver(const Solver &solver) const {
            // x = Gaussian
            int ncol = 1;
            std::shared_ptr<CoarseSpinor> x =
                AuxC::tmp(*_mg_levels->coarse_levels.back().info, ncol);
            Gaussian(*x);

            //
            // Compute V'*A*V*x in one way
            //

            // Transfer to the first coarse level
            std::shared_ptr<CoarseSpinor> coarse_out = x;
            for (int coarse_idx = (int)_mg_levels->coarse_levels.size() - 2; coarse_idx >= 0;
                 coarse_idx--) {
                std::shared_ptr<CoarseSpinor> out_level =
                    AuxC::tmp(*_mg_levels->coarse_levels[coarse_idx].info, ncol);
                _Transfer_coarse_level[coarse_idx]->P(*coarse_out, *out_level);
                coarse_out = out_level;
            }

            // Transfer to the fine level
            std::shared_ptr<QPhiXSpinorF> out_f = AuxQ::tmp(*_mg_levels->fine_level.info, ncol);
            _Transfer_fine_level->P(*coarse_out, *out_f);
            coarse_out.reset();

            // Ax = A*out_f
            std::shared_ptr<QPhiXSpinorF> Ax = AuxQ::tmp(*_mg_levels->fine_level.info, ncol);
            ZeroVec(*Ax, SUBSET_ALL);
            _M_fine->unprecOp(*Ax, *out_f, LINOP_OP);
            out_f.reset();

            // Transfer to first coarse level
            std::shared_ptr<CoarseSpinor> coarse_in =
                AuxC::tmp(*_mg_levels->coarse_levels[0].info, ncol);
            _Transfer_fine_level->R(*Ax, *coarse_in);
            Ax.reset();

            // Transfer to the deepest level
            for (unsigned int coarse_idx = 1; coarse_idx < _mg_levels->coarse_levels.size();
                 coarse_idx++) {
                std::shared_ptr<CoarseSpinor> in_level =
                    AuxC::tmp(*_mg_levels->coarse_levels[coarse_idx].info, ncol);
                _Transfer_coarse_level[coarse_idx - 1]->R(*coarse_in, *in_level);
                coarse_in = in_level;
            }

            //
            // Compute V'*A*V*x throw the coarse operator and compare
            //

            // y = (V'AV)*x
            std::shared_ptr<CoarseSpinor> y =
                AuxC::tmp(*_mg_levels->coarse_levels.back().info, ncol);
            _mg_levels->coarse_levels.back().M->unprecOp(*y, *x);
            double n_VAVx = Norm2Vec(*coarse_in)[0];
            double n_diff_mv = XmyNorm2Vec(*y, *coarse_in)[0];
            
            // y = solve(coarse_in)
            ZeroVec(*y);
            solver(*y, *coarse_in);

            double n_x = Norm2Vec(*x)[0];
            double n_diff = XmyNorm2Vec(*y, *x)[0];
            MasterLog(INFO, "MGDeflation: coarsest operator error %g  solver error %g",
                      n_diff_mv / n_VAVx, n_diff / n_x);
        }

        const std::shared_ptr<LatticeInfo> _info;
        const std::shared_ptr<const QPhiXWilsonCloverEOLinearOperatorF> _M_fine;
        std::shared_ptr<QPhiXMultigridLevelsEO> _mg_levels;
        std::shared_ptr<CoarseSpinor> _eigenvectors; // eigenvectors of \gamma_5 * inv(A)
        std::vector<float> _eigenvalues;             // eigenvalues of \gamma_5 * inv(A)
        std::shared_ptr<QPhiXTransfer<QPhiXSpinorF>> _Transfer_fine_level;
        std::vector<std::shared_ptr<CoarseTransfer>> _Transfer_coarse_level;
    };
}

#endif // INCLUDE_LATTICE_QPHIX_QPHIX_DISCO_H_
