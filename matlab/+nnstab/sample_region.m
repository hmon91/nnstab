function X = sample_region(r, k, seed)
%SAMPLE_REGION  k starting states drawn evenly (uniformly) from a certified region.
%   X = nnstab.sample_region(chk, k)
%   X = nnstab.sample_region(chk, k, seed)     (default seed = 0)
%
%   chk is a certified result of nnstab.check_local, or a result of
%   nnstab.verify with status 'local'. X is n x k, one state per column, in
%   the region {x >= 0 : v'*x <= rho}. The random number generator is
%   restored afterwards.

    if nargin < 3, seed = 0; end
    if isfield(r, 'status'), r = r.local_check; end     % a result of nnstab.verify
    if isempty(r) || ~isfield(r, 'rho') || ~r.certified
        error('nnstab:sample_region:certified', ...
              'The local check is not certified, so there is no region to sample.');
    end
    state = rng;
    rng(seed);
    E = -log(rand(numel(r.v) + 1, k));       % exponential samples
    rng(state);
    U = E ./ sum(E, 1);                      % uniform on the simplex, with one slack entry
    X = r.rho * U(1:end-1, :) ./ r.v;
end
