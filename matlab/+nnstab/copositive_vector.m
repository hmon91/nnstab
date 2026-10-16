function [v, d] = copositive_vector(M)
%COPOSITIVE_VECTOR  For a Metzler, Hurwitz M: v > 0 and a rate d > 0 with v'*M <= -d*v'.
%   [v, d] = nnstab.copositive_vector(M)
%
%   v is the left Perron vector of M (smallest entry 1), which gives the
%   best rate, d = -spectral_abscissa(M). If that vector is not strictly
%   positive (this can happen when M is reducible), the certificate
%   (-M')\1 is used instead.

    v = nnstab.perron_vector(M);
    if ~isempty(v)
        d = min(-(M.'*v) ./ v);
        if d > 0
            return
        end
    end
    v = nnstab.hurwitz_certificate(M);
    if isempty(v)
        error('nnstab:copositive_vector:matrix', 'M must be Metzler and Hurwitz.');
    end
    v = v/min(v);
    d = min(-(M.'*v) ./ v);
end
