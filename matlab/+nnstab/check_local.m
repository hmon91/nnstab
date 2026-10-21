function c = check_local(A, B, C, net, ybar, varargin)
%CHECK_LOCAL  Local test on the output box 0 <= y <= ybar, with its region of attraction.
%   c = nnstab.check_local(A, B, C, net, ybar)
%   c = nnstab.check_local(..., 'v', 'lp', 'MinDecay', 0.5)
%
%   gamma1 and gamma2 come from nnsector.local_bound on the box, about
%   y* = 0. Certified if A + B*gamma1*C is Metzler and A + B*gamma2*C is
%   Hurwitz. Then the region {x >= 0 : v'*x <= rho} is forward invariant,
%   and every solution that starts in it satisfies
%       v'*x(t)   <= exp(-decay*t) * v'*x0,
%       sum(x(t)) <= overshoot * exp(-decay*t) * sum(x0).
%   rho is the largest level with C*x <= ybar on the region. ybar is a
%   positive scalar (used for every output) or a vector with one entry per
%   output.
%
%   Options
%     'v'         'perron' (default): the left eigenvector of A + B*gamma2*C
%                 for its largest eigenvalue, which gives the fastest
%                 guaranteed decay. 'lp': the v that makes the region as
%                 large as possible while the decay rate stays at least
%                 MinDecay (or, if MinDecay is 0, half the best rate).
%     'MinDecay'  smallest decay rate to accept (default 0: any rate)
%
%   Fields: ybar, gamma1, gamma2, M1, M2, metzler, hurwitz, certified,
%   decay, v (smallest entry 1), rho, radius (every x0 >= 0 with
%   sum(x0) <= radius is in the region), overshoot = max(v)/min(v), method
%   and reason (why the test failed, '' if it passed).

    ip = inputParser;
    ip.FunctionName = 'nnstab.check_local';
    ip.addParameter('v', 'perron', @ischar);
    ip.addParameter('MinDecay', 0, @(d) isnumeric(d) && isscalar(d) && isreal(d));
    ip.parse(varargin{:});
    method    = lower(ip.Results.v);
    min_decay = double(ip.Results.MinDecay);

    [A, B, C, net] = nnstab.check_system(A, B, C, net);
    p = net.p;
    ybar = double(ybar(:));
    if isscalar(ybar), ybar = repmat(ybar, p, 1); end
    if numel(ybar) ~= p || any(~isfinite(ybar)) || any(ybar <= 0)
        error('nnstab:check_local:ybar', ...
              'ybar must be a positive scalar or have %d positive entries.', p);
    end
    if ~any(strcmp(method, {'perron', 'lp'}))
        error('nnstab:check_local:v', 'v must be ''perron'' or ''lp''.');
    end
    if ~(min_decay >= 0)
        error('nnstab:check_local:min_decay', 'MinDecay must be >= 0.');
    end

    sb = nnsector.local_bound(net, zeros(p, 1), ybar, 'ystar', zeros(p, 1));
    g1 = sb.gamma1;
    g2 = sb.gamma2;
    M1 = A + B*g1*C;
    M2 = A + B*g2*C;
    c = struct('ybar', ybar, 'gamma1', g1, 'gamma2', g2, 'M1', M1, 'M2', M2, ...
               'metzler', nnstab.is_metzler(M1), 'hurwitz', nnstab.is_hurwitz(M2), ...
               'certified', false, 'decay', 0, 'v', [], 'rho', [], 'radius', 0, ...
               'overshoot', [], 'method', method, 'reason', '');
    if ~c.metzler
        c.reason = ['A + B gamma1 C is not Metzler: the lower slope gamma1 is too negative ' ...
                    'to keep the states nonnegative.'];
        return
    end
    if ~c.hurwitz
        c.reason = 'A + B gamma2 C is not Hurwitz: the upper slope gamma2 is too large.';
        return
    end
    best = -nnstab.spectral_abscissa(M2);
    if min_decay > 0 && min_decay >= best
        c.reason = sprintf('The best possible decay rate here is %.3g, not above MinDecay = %g.', ...
                           best, min_decay);
        return
    end

    if strcmp(method, 'perron')
        [v, decay] = nnstab.copositive_vector(M2);
        kappa = max(C ./ v.', [], 2);           % largest C(j,i)/v(i), per output j
        hit = kappa > 0;
        if any(hit)
            rho = min(ybar(hit) ./ kappa(hit));
        else
            rho = Inf;
        end
    else
        if min_decay > 0, rate = min_decay; else, rate = 0.5*best; end
        cv = max(C ./ ybar, [], 1).';           % C*x <= ybar on the region if w >= cv
        cv = max(cv, 1e-12*max(max(cv), 1));    % keep w > 0 so the region is bounded
        w = nnstab.largest_region_vector(M2, cv, rate);
        decay = min(-(M2.'*w) ./ w);
        v   = w/min(w);
        rho = 1/min(w);
    end
    if ~(decay > 0) || (min_decay > 0 && decay < min_decay*(1 - 1e-9))
        c.reason = 'No copositive vector with the requested decay rate was found.';
        return
    end
    c.certified = true;
    c.decay     = decay;
    c.v         = v;
    c.rho       = rho;
    c.radius    = rho/max(v);
    c.overshoot = max(v)/min(v);
end
