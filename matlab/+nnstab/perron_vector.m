function v = perron_vector(M)
%PERRON_VECTOR  Left eigenvector of a Metzler M for its eigenvalue with the largest real part.
%   v = nnstab.perron_vector(M)
%
%   Scaled so that its smallest entry is 1. It is [] if the vector is not
%   strictly positive (this can happen when M is reducible).

    [V, D] = eig(M.');
    [~, k] = max(real(diag(D)));
    v = real(V(:, k));
    [~, j] = max(abs(v));
    v = v*sign(v(j));
    if any(~isfinite(v)) || min(v) <= 1e-12*max(v)
        v = [];
        return
    end
    v = v/min(v);
end
