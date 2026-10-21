function res = verify(A, B, C, net, varargin)
%VERIFY  Check exponential stability of x' = A x + B pi(C x) for x0 >= 0.
%   res = nnstab.verify(A, B, C, net)
%   res = nnstab.verify(..., 'ybar', ybar)          check only this output box
%   res = nnstab.verify(..., 'Direction', d)        shape of the searched boxes
%   res = nnstab.verify(..., 'v', 'lp', 'MinDecay', 0.5, 'RelTol', 1e-3)
%
%   First the global test (nnstab.check_global). If it fails, the local
%   test (nnstab.check_local) on the output boxes 0 <= y <= s*d, where d is
%   all ones by default: s is doubled (or halved) until the test changes
%   its answer, the edge is found by bisection to relative accuracy RelTol,
%   and the largest region among a few boxes below the edge is kept. With
%   'ybar', only that box is checked. 'v' and 'MinDecay' are passed to
%   nnstab.check_local. Near the edge of the largest region the guaranteed
%   decay rate gets small; ask for a faster one with 'MinDecay'.
%
%   res.status is 'global' (exponentially stable for every x0 >= 0),
%   'local' (exponentially stable on the region of res.local_check) or
%   'none' (neither test passed; this does not mean the loop is unstable).
%   Other fields: stable, global_check, local_check ([] if the global test
%   passed), boxes_tried and note. nnstab.report(res) prints a report, and
%   nnstab.in_region(res, x0) tells if a starting state is covered.

    ip = inputParser;
    ip.FunctionName = 'nnstab.verify';
    ip.addParameter('ybar', [], @isnumeric);
    ip.addParameter('Direction', [], @isnumeric);
    ip.addParameter('v', 'perron', @ischar);
    ip.addParameter('MinDecay', 0, @(d) isnumeric(d) && isscalar(d) && isreal(d));
    ip.addParameter('RelTol', 1e-3, @(r) isnumeric(r) && isscalar(r) && r > 0);
    ip.parse(varargin{:});
    opt = ip.Results;

    [A, B, C, net] = nnstab.check_system(A, B, C, net);
    res = struct('status', 'none', 'stable', false, 'global_check', [], ...
                 'local_check', [], 'boxes_tried', 0, 'note', '');
    res.global_check = nnstab.check_global(A, B, C, net);
    if res.global_check.certified
        res.status = 'global';
        res.stable = true;
        return
    end
    check_box = @(ybar) nnstab.check_local(A, B, C, net, ybar, 'v', opt.v, 'MinDecay', opt.MinDecay);
    if ~isempty(opt.ybar)
        res.local_check = check_box(opt.ybar);
        res.boxes_tried = 1;
    else
        d = opt.Direction;
        if isempty(d), d = ones(net.p, 1); end
        d = double(d(:));
        if numel(d) ~= net.p || any(~isfinite(d)) || any(d <= 0)
            error('nnstab:verify:direction', 'Direction must have %d positive entries.', net.p);
        end
        [res.local_check, res.boxes_tried, res.note] = search(@(s) check_box(s*d), opt.RelTol);
        if res.local_check.certified && isempty(res.note) && opt.MinDecay == 0
            res.note = ['this is the largest region found, and near its edge the guaranteed ' ...
                        'decay rate is small; pass MinDecay for a faster guaranteed rate on a ' ...
                        'smaller region.'];
        end
    end
    if res.local_check.certified
        res.status = 'local';
        res.stable = true;
    end
end

% -------------------------------------------------------------------------
function [best, tried, note] = search(check, rtol)
% Find the box scale s with the largest certified region.
    max_scale = 2^40;
    min_scale = 2^-40;
    tried = 0;
    note  = '';
    s = 1;
    first = check(s);  tried = tried + 1;
    if first.certified
        lo = s;  lo_chk = first;  hi = [];
        while s < max_scale
            s = 2*s;
            c = check(s);  tried = tried + 1;
            if ~c.certified
                hi = s;
                break
            end
            lo = s;  lo_chk = c;
        end
        if isempty(hi)
            best = lo_chk;
            note = ['the local test still passes on very large boxes; the loop may be ' ...
                    'globally stable, but the global test is too conservative.'];
            return
        end
    else
        hi = s;  lo_chk = [];  last = first;
        while s > min_scale
            s = s/2;
            c = check(s);  tried = tried + 1;
            if c.certified
                lo = s;  lo_chk = c;
                break
            end
            hi = s;  last = c;
        end
        if isempty(lo_chk)
            best = last;
            note = 'the local test fails even on very small boxes.';
            return
        end
    end
    while hi - lo > rtol*hi
        mid = 0.5*(lo + hi);
        c = check(mid);  tried = tried + 1;
        if c.certified
            lo = mid;  lo_chk = c;
        else
            hi = mid;
        end
    end
    best = lo_chk;
    scales = lo*linspace(0.5, 1, 6);
    for s = scales(1:end-1)                  % the region need not be largest at the edge
        c = check(s);  tried = tried + 1;
        if c.certified && c.radius > best.radius
            best = c;
        end
    end
end
