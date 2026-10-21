function tf = is_metzler(M, tol)
%IS_METZLER  True if every entry of M off the diagonal is >= 0.
%   tf = nnstab.is_metzler(M)
%   tf = nnstab.is_metzler(M, tol)
%
%   Entries down to -tol*max(1, max(abs(M(:)))) count as >= 0, for rounding
%   (default tol = 1e-12).

    if nargin < 2, tol = 1e-12; end
    off = M - diag(diag(M));
    tf = all(off(:) >= -tol*max(1, max(abs(M(:)))));
end
