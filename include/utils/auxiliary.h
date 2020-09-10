#ifndef INCLUDE_LATTICE_AUXILIARY_H
#define INCLUDE_LATTICE_AUXILIARY_H

#include <algorithm>
#include <cmath>
#include <lattice/lattice_info.h>
#include <memory>
#include <vector>

namespace MG {
    namespace aux {
        template <typename T> std::vector<T> sqrt(const std::vector<T> &v) {
            std::vector<T> out(v.size());
            std::transform(v.begin(), v.end(), out.begin(),
                           [](const T &x) { return std::sqrt(x); });
            return out;
        }

        template <typename T>
        std::vector<T> operator/(const std::vector<T> &a, const std::vector<T> &b) {
            assert(a.size() == b.size());
            std::vector<T> out(a.size());
            std::transform(a.begin(), a.end(), b.begin(), out.begin(),
                           [](const T &x, const T &y) { return x / y; });
            return out;
        }
    }

    template <typename Spinor> class AbstractSpinor {
    public:
        // virtual bool is_like(const Spinor& s) const = 0;
        // virtual bool is_like(const LatticeInfo& info, int ncol) const = 0;
    };

    template <typename Spinor> class AuxiliarySpinors {
    public:
        AuxiliarySpinors() : subrogate(nullptr) {}
        AuxiliarySpinors(const AuxiliarySpinors<Spinor> *subrogate_) : subrogate(subrogate_) {}

        // Subrogate the calls to another instance
        void subrogateTo(AuxiliarySpinors<Spinor> *a) {
            subrogateTo((const AuxiliarySpinors<Spinor> *)a);
        }

        void subrogateTo(const AuxiliarySpinors<Spinor> *a) {
            a->_tmp.insert(a->_tmp.end(), _tmp.begin(), _tmp.end());
            _tmp.clear();
            subrogate = a;
        }

        // Return a spinor with a shape like the given one
        std::shared_ptr<Spinor> tmp(const LatticeInfo &info, int ncol) const {
            if (subrogate) return subrogate->tmp(info, ncol);

            std::shared_ptr<Spinor> s;

            // Find a spinor not being used
            for (auto it = _tmp.begin(); it != _tmp.end(); it++) {
                if (it->use_count() <= 1) {
                    // If the free spinor is not like the given
                    // one, replace it with a new one.
                    if (!it->get()->is_like(info, ncol)) { it->reset(new Spinor(info, ncol)); }
                    s = *it;
                    break;
                }
            }

            // If every spinor is busy, create a new one
            if (!s) {
                s.reset(new Spinor(info, ncol));
                _tmp.emplace_back(s);
            }
            return s;
        }

        // Return a spinor with a shape like the given one
        std::shared_ptr<Spinor> tmp(const Spinor &like_this) const {
            return tmp(like_this.GetInfo(), like_this.GetNCol());
        }

        // Remove unused Spinors
        void clear() const {
            if (subrogate) return subrogate->clear();

            // Remove spinors not being used
            unsigned int n = 0;
            for (auto it = _tmp.begin(); it != _tmp.end(); it++) {
                if (it->use_count() <= 1) {
                    it->reset();
                } else {
                    _tmp[n++] = *it;
                }
            }
            _tmp.resize(n);
        }

    private:
        mutable std::vector<std::shared_ptr<Spinor>> _tmp;
        const AuxiliarySpinors<Spinor> *subrogate;
    };


    /**
     * Simple wrapper around a type
     *
     * \param T: Type to wrapper
     * \param UniqueId: optional identifier
     *
     * Example for initializing derived members before base classes:
     *
     * ```
     * class B : private Box<A>, private Box<A,1>, C {
     *    public:
     *    B(int i) : Box<A>{A(i)}, Box<A,1>{A(i*2)}, C(Box<A>::member, Box<A,1>::member) {}
     * };
     * ```
     */

    template <typename T, int UniqueId = 0> struct Box {
        using member_type = T;
        static constexpr int uniqueId = UniqueId;
        T member;
    };
}

#endif
