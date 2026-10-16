function tf = is_hurwitz(M)
%IS_HURWITZ  True if every eigenvalue of M has a negative real part.
%   tf = nnstab.is_hurwitz(M)
%
%   For a Metzler M the answer comes from nnstab.hurwitz_certificate, which
%   is a direct test and does not rely on computed eigenvalues.

    if nnstab.is_metzler(M)
        tf = ~isempty(nnstab.hurwitz_certificate(M));
    else
        tf = nnstab.spectral_abscissa(M) < 0;
    end
end
