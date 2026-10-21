function tf = in_region(r, X, tol)
%IN_REGION  True if a result covers every starting state in X.
%   tf = nnstab.in_region(res, X)    res from nnstab.verify
%   tf = nnstab.in_region(chk, X)    chk from nnstab.check_local
%
%   X holds one starting state per column (n x N); one state may also be a
%   row. For a global result every x0 >= 0 is covered; for a local one,
%   the states in the region {x >= 0 : v'*x <= rho}. tol (default 1e-12)
%   allows for rounding.

    if nargin < 3, tol = 1e-12; end
    if isfield(r, 'status')                    % a result of nnstab.verify
        switch r.status
            case 'global'
                tf = all(X(:) >= -tol);
                return
            case 'local'
                r = r.local_check;
            otherwise
                tf = false;
                return
        end
    end
    if isempty(r) || ~isfield(r, 'rho') || ~r.certified
        tf = false;
        return
    end
    n = numel(r.v);
    if isvector(X) && numel(X) == n, X = X(:); end
    tf = all(X(:) >= -tol) && all(r.v.'*X <= r.rho*(1 + tol) + tol);
end
