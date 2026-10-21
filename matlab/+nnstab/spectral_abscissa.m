function a = spectral_abscissa(M)
%SPECTRAL_ABSCISSA  Largest real part of the eigenvalues of M.
%   a = nnstab.spectral_abscissa(M)

    a = max(real(eig(M)));
end
