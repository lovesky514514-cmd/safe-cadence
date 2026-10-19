"""IEEE-754 outward-rounded enclosures for squared Euclidean distances.

The intervals enclose exact real arithmetic on the *stored binary64 query
coordinates and entity coordinates*, not the real-valued h+r expression before
float evaluation. The query construction uses NumPy float64, by definition.

Assumptions: IEEE-754 correctly rounded +,-,*, finite inputs and results,
no underflow-to-zero mode, NumPy nextafter honors IEEE-754. Fail closed if
nonfinite intermediate results occur. These are implementation assumptions,
not a language-independent formal floating point verification.
"""
from __future__ import annotations
import numpy as np

NEG=-np.inf
POS=np.inf


def sqdistance_bounds(q: np.ndarray, candidates: np.ndarray):
    """Return (lo,hi), enclosing exact squared norms, each in [0,+inf].

    q: shape (d,), candidates: (n,d). Bounds are binary64 and outward rounded.
    """
    q=np.asarray(q,dtype=np.float64)
    x=np.asarray(candidates,dtype=np.float64)
    if x.ndim!=2 or q.shape!=(x.shape[1],):raise ValueError('shape mismatch')
    if not np.isfinite(q).all() or not np.isfinite(x).all():raise ValueError('nonfinite input')
    lo=np.zeros(len(x),dtype=np.float64)
    hi=np.zeros(len(x),dtype=np.float64)
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        for j in range(x.shape[1]):
            diff=q[j]-x[:,j]
            a=np.nextafter(diff,NEG)
            b=np.nextafter(diff,POS)
            smallest=np.where(a>0,a,np.where(b<0,-b,0.))
            largest=np.maximum(np.abs(a),np.abs(b))
            lsq=np.maximum(0.,np.nextafter(smallest*smallest,NEG))
            usq=np.nextafter(largest*largest,POS)
            lo=np.maximum(0.,np.nextafter(lo+lsq,NEG))
            hi=np.nextafter(hi+usq,POS)
    if not np.isfinite(hi).all():
        # Infinite upper bound means the caller cannot certify on this batch.
        hi=np.where(np.isfinite(hi),hi,np.inf)
    return lo,hi


def upper_distance(hi_sq):
    with np.errstate(over='ignore',invalid='ignore'):
        return np.nextafter(np.sqrt(hi_sq),POS)


def lower_distance(lo_sq):
    with np.errstate(over='ignore',invalid='ignore'):
        return np.maximum(0.,np.nextafter(np.sqrt(lo_sq),NEG))


def sqdistance_bounds_fast(q, candidates):
    """Vectorized conservative enclosure when square products are normal.

    If every nonzero subtraction and square is normal and no overflow occurs,
    IEEE fl operations each contribute relative error <= u. An inflated
    16*(d+4)*eps budget dominates the (d+3)-operation gamma bound for
    dimensions supported here. Near underflow the strict nextafter-based
    interval routine is used instead, independently for the affected rows.

    This is a floating-point argument conditional on IEEE behavior and is
    NOT a verified C implementation or hardware-independent formal proof.
    """
    q=np.asarray(q,dtype=np.float64)
    x=np.asarray(candidates,dtype=np.float64)
    if x.ndim!=2 or q.shape!=(x.shape[1],):raise ValueError('shape mismatch')
    if not np.isfinite(q).all() or not np.isfinite(x).all():raise ValueError('nonfinite input')
    d=x.shape[1]
    eps=np.finfo(np.float64).eps
    inflate=16.0*(d+4)*eps
    if inflate>=.1:return sqdistance_bounds(q,x)
    with np.errstate(all='ignore'):
        dif=q[None,:]-x
        s=np.einsum('ij,ij->i',dif,dif)
        lo=np.maximum(0.,np.nextafter(s*(1.-inflate),-np.inf))
        hi=np.nextafter(s*(1.+inflate),np.inf)
        min_normal_sq=np.sqrt(np.finfo(np.float64).tiny)
        anomalous=(~np.isfinite(s)) | (~np.isfinite(dif).all(axis=1)) | np.any((np.abs(dif)>0)&(np.abs(dif)<min_normal_sq),axis=1)
        if np.any(anomalous):
            slo,shi=sqdistance_bounds(q,x[anomalous])
            lo[anomalous]=slo;hi[anomalous]=shi
    return lo,hi


def sqdistance_pairs_fast(queries, entities):
    """Conservative squared-distance bounds for broadcast-compatible (...,d)."""
    q=np.asarray(queries,dtype=np.float64)
    x=np.asarray(entities,dtype=np.float64)
    if q.shape[-1]!=x.shape[-1]:raise ValueError('dimension mismatch')
    if not np.isfinite(q).all() or not np.isfinite(x).all():raise ValueError('nonfinite input')
    d=q.shape[-1]; eps=np.finfo(np.float64).eps
    inflation=16.*(d+4)*eps
    with np.errstate(all='ignore'):
        dif=q-x
        s=np.einsum('...i,...i->...',dif,dif)
        lo=np.maximum(0.,np.nextafter(s*(1-inflation),-np.inf))
        hi=np.nextafter(s*(1+inflation),np.inf)
        tiny=np.sqrt(np.finfo(np.float64).tiny)
        bad=(~np.isfinite(s))|np.any((np.abs(dif)>0)&(np.abs(dif)<tiny),axis=-1)
    if np.any(bad):
        q2,x2=np.broadcast_arrays(q,x)
        for ij in np.argwhere(bad):
            idx=tuple(ij)
            zlo,zhi=sqdistance_bounds(q2[idx],x2[idx][None,:])
            lo[idx]=zlo[0];hi[idx]=zhi[0]
    return lo,hi


def sqdistance_pairs_outward(queries,entities):
    """Strict operation-by-operation outward rounding (slower than fast gamma).

    Every subtraction, square and accumulation is expanded with np.nextafter.
    This path provides an explicit IEEE-754 interval certificate for distances
    between stored binary64 coordinates; it does not require normal products.
    """
    q,x=np.broadcast_arrays(np.asarray(queries,dtype=np.float64),
                            np.asarray(entities,dtype=np.float64))
    shape=q.shape[:-1];d=q.shape[-1]
    q=q.reshape(-1,d);x=x.reshape(-1,d)
    lo=np.empty(len(x),dtype=np.float64);hi=np.empty(len(x),dtype=np.float64)
    # The strict arithmetic kernel itself is NumPy-vectorized across pairs.
    with np.errstate(all='ignore'):
        lo.fill(0);hi.fill(0)
        for j in range(d):
            diff=q[:,j]-x[:,j]
            a=np.nextafter(diff,-np.inf);b=np.nextafter(diff,np.inf)
            small=np.where(a>0,a,np.where(b<0,-b,0.))
            large=np.maximum(np.abs(a),np.abs(b))
            sqlo=np.maximum(0.,np.nextafter(small*small,-np.inf))
            sqhi=np.nextafter(large*large,np.inf)
            lo=np.maximum(0.,np.nextafter(lo+sqlo,-np.inf))
            hi=np.nextafter(hi+sqhi,np.inf)
    return lo.reshape(shape),hi.reshape(shape)
