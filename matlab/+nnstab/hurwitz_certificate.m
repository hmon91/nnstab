function v = hurwitz_certificate(M)
%HURWITZ_CERTIFICATE  For a Metzler M: a vector v > 0 with v'*M = -1', or [] if there is none.
%   v = nnstab.hurwitz_certificate(M)
%
%   Such a v exists exactly when M is Hurwitz, since then -M is a
%   nonsingular M-matrix and v = (-M')\1 > 0. The result is checked, so a
%   returned v is a certificate on its own.

    Mt = -M.';
    v  = [];
    if rcond(Mt) < eps
        return
    end
    w = Mt \ ones(size(M, 1), 1);
    if all(isfinite(w)) && all(w > 0) && all(M.'*w < 0)
        v = w;
    end
end
