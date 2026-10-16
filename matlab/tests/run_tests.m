function run_tests()
%RUN_TESTS  Tests for the nnstab library.
%   Run with nnstab/matlab/tests as the current folder (or on the path):
%       run_tests
%   It needs the MATLAB version of nnsector: set the environment variable
%   NNSECTOR_MATLAB to its folder (the one that contains +nnsector), put the
%   nnsector repository next to the nnstab repository, or add it to the
%   path yourself. Prints PASS or FAIL for each test and ends with an error
%   if any failed. Some tests compare with numbers from the Python package.

    here = fileparts(mfilename('fullpath'));
    root = fileparts(here);
    addpath(root);
    add_nnsector(root);

    st = rng;  rng(0);
    restore = onCleanup(@() rng(st));   %#ok<NASGU>
    nfail = 0;

    % ---- matrix tools -------------------------------------------------------
    ok = true;
    for t = 1:50
        M  = random_metzler_hurwitz(randi(6));
        n  = size(M, 1);
        Mu = M + (0.01 - nnstab.spectral_abscissa(M))*eye(n);      % not Hurwitz
        v  = nnstab.hurwitz_certificate(M);
        p  = nnstab.perron_vector(M);
        [vc, d] = nnstab.copositive_vector(M);
        ok = ok && ~isempty(v) && all(v > 0) && near(M.'*v, -ones(n, 1), 1e-9) && ...
             isempty(nnstab.hurwitz_certificate(Mu)) && nnstab.is_hurwitz(M) && ~nnstab.is_hurwitz(Mu) && ...
             ~isempty(p) && abs(min(p) - 1) < 1e-12 && ...
             near(M.'*p, nnstab.spectral_abscissa(M)*p, 1e-9*max(p)) && ...
             abs(d + nnstab.spectral_abscissa(M)) < 1e-9 && all(M.'*vc <= -d*vc*(1 - 1e-9));
    end
    nfail = nfail + tally('random Metzler, Hurwitz matrices: certificate, Perron vector, copositive vector', ok);

    nfail = nfail + tally('is_metzler allows only rounding-size negatives', ...
        nnstab.is_metzler([-1 0; 2 -3]) && ~nnstab.is_metzler([-1 -0.1; 2 -3]) && ...
        nnstab.is_metzler([5 1e-20; -1e-14 3]));
    nfail = nfail + tally('is_hurwitz on matrices that are not Metzler', ...
        nnstab.is_hurwitz([-1 -5; 5 -1]) && ~nnstab.is_hurwitz([1 -5; 5 1]));      % -1 +- 5i and 1 +- 5i

    Mr = [-1 0; 1 -2];                  % reducible: its Perron vector has a zero entry
    [v, d] = nnstab.copositive_vector(Mr);
    nfail = nfail + tally('copositive_vector of a reducible matrix, and of a matrix that is not Hurwitz', ...
        all(v > 0) && all(Mr.'*v <= -d*v) && d > 0 && d <= 1 && ...
        raises(@() nnstab.copositive_vector([1 0; 0 -1]), 'nnstab:copositive_vector:matrix'));

    ok = true;
    for t = 1:40
        n = randi(6);
        M = random_metzler_hurwitz(n);
        c = abs(randn(n, 1)).*(rand(n, 1) < 0.8);
        rate = 0.9*rand*(-nnstab.spectral_abscissa(M));
        w  = nnstab.largest_region_vector(M, c, rate);
        s  = (M + rate*eye(n)).'*w;
        sc = 1 + max(abs(w));
        p  = nnstab.perron_vector(M);
        tp = max(1, max(c./p));         % a scaled Perron vector is feasible, so it is >= w
        ok = ok && all(w >= c - 1e-12*sc) && all(s <= 1e-10*sc) && ...
             all(abs((w - c).*s) <= 1e-9*sc^2) && all(tp*p >= w - 1e-9*sc);
    end
    nfail = nfail + tally('largest_region_vector is feasible, complementary and the least', ok);

    % ---- the global test ----------------------------------------------------
    [A, B, C, net] = nnstab.example_system('global');
    g = nnstab.check_global(A, B, C, net);
    nfail = nfail + tally('global example: certified, Gamma2 = 4.7914', ...
        g.certified && g.metzler && g.hurwitz && isempty(g.reason) && ...
        near(g.Gamma2, 4.79136, 1e-5) && isequal(g.Gamma1, -g.Gamma2) && ...
        abs(g.decay + nnstab.spectral_abscissa(g.M2)) < 1e-9 && all(g.M2.'*g.v <= -g.decay*g.v*(1 - 1e-9)));
    nfail = nfail + tally('global example: the same numbers as the Python package', ...
        rel(g.Gamma2, 4.7913557794152739) && rel(g.decay, 0.46752964815681031) && ...
        rel(g.overshoot, 2.32643097521933));

    [A, B, C, net] = nnstab.example_system('local');
    g = nnstab.check_global(A, B, C, net);
    nfail = nfail + tally('local example: the global test fails (not Metzler)', ...
        ~g.certified && ~g.metzler && contains_text(g.reason, 'Metzler') && ...
        isempty(g.overshoot) && g.decay == 0);

    % ---- the local test -----------------------------------------------------
    c = nnstab.check_local(A, B, C, net, 1);
    corners = diag(c.rho./c.v);         % the corners of the region, one per column
    nfail = nfail + tally('local example on 0 <= y <= 1: certified, the same numbers as Python', ...
        c.certified && strcmp(c.method, 'perron') && ...
        near(c.gamma1, -2.86175, 1e-5) && near(c.gamma2, -1.14397, 1e-5) && ...
        rel(c.rho, 1.9118059689214806) && rel(c.decay, 1.3306820894165547) && ...
        rel(c.v, [1.9118059689214808; 2.1165260164243711; 1]) && ...
        abs(c.decay + nnstab.spectral_abscissa(c.M2)) < 1e-9);
    nfail = nfail + tally('local example: C*x <= ybar on the region, reached at a corner', ...
        all(C*corners <= 1 + 1e-12) && abs(max(C*corners) - 1) < 1e-12 && ...
        nnstab.in_region(c, 0.5*corners) && ~nnstab.in_region(c, 1.01*corners(:, 1)) && ...
        ~nnstab.in_region(c, [-0.1; 0; 0]));

    lp  = nnstab.check_local(A, B, C, net, 1, 'v', 'lp', 'MinDecay', 0.999*c.decay);
    lph = nnstab.check_local(A, B, C, net, 1, 'v', 'lp');
    nfail = nfail + tally('v = ''lp'': the region reaches at least as far in every direction', ...
        lp.certified && lp.decay >= 0.999*c.decay*(1 - 1e-9) && all(lp.rho./lp.v >= c.rho./c.v*(1 - 1e-9)) && ...
        lph.decay >= 0.5*c.decay*(1 - 1e-9) && all(lph.rho./lph.v >= lp.rho./lp.v*(1 - 1e-9)));

    c3 = nnstab.check_local(A, B, C, net, 3);
    nfail = nfail + tally('local example on 0 <= y <= 3: not certified, not Hurwitz', ...
        ~c3.certified && c3.metzler && ~c3.hurwitz && contains_text(c3.reason, 'Hurwitz') && ...
        c3.radius == 0 && ~nnstab.in_region(c3, zeros(3, 1)));

    cs = nnstab.check_local(A, B, C, net, 1, 'MinDecay', 0.9*c.decay);
    cm = nnstab.check_local(A, B, C, net, 1, 'MinDecay', 1.1*c.decay);
    nfail = nfail + tally('MinDecay: accepted below the best rate, refused above it', ...
        cs.certified && ~cm.certified && contains_text(cm.reason, 'MinDecay'));

    ok = true;
    for t = 1:40                        % no hidden layer: pi(y) = K*y, the sector is [K, K]
        Ar = abs(randn(3)) - 2.5*eye(3);
        Br = abs(randn(3, 1));
        Cr = abs(randn(1, 3));
        K  = -1.5 + 2*rand;
        cr = nnstab.check_local(Ar, Br, Cr, nnsector.network({K}), 1);
        M  = Ar + K*Br*Cr;
        ok = ok && cr.certified == (nnstab.is_metzler(M) && nnstab.spectral_abscissa(M) < 0);
    end
    nfail = nfail + tally('a linear controller gives the linear test', ok);

    cv = nnstab.check_local(A, B.', C.', net, 1);        % B as a row, C as a column
    nfail = nfail + tally('B and C may be given as vectors of either shape', ...
        isequal(cv.v, c.v) && cv.rho == c.rho);

    nfail = nfail + tally('B < 0, C < 0, biases and sigmoid are errors', ...
        raises(@() nnstab.check_global(A, -B, C, net), 'nnstab:system:positive') && ...
        raises(@() nnstab.check_global(A, B, -C, net), 'nnstab:system:positive') && ...
        raises(@() nnstab.check_global(A, B, C, nnsector.network(net.W, {[], [], 0.1})), ...
               'nnstab:system:equilibrium') && ...
        raises(@() nnstab.check_global(A, B, C, nnsector.network(net.W, [], 'Activation', 'sigmoid')), ...
               'nnstab:system:equilibrium'));
    nfail = nfail + tally('sizes that do not fit and bad options are errors', ...
        raises(@() nnstab.check_global(A, ones(3, 2), C, net), 'nnstab:system:size') && ...
        raises(@() nnstab.check_global(A, B, ones(2, 3), net), 'nnstab:system:size') && ...
        raises(@() nnstab.check_global(ones(3, 2), B, C, net), 'nnstab:system:size') && ...
        raises(@() nnstab.check_global(A, B, C, 'not a network'), 'nnstab:system:net') && ...
        raises(@() nnstab.check_local(A, B, C, net, 0), 'nnstab:check_local:ybar') && ...
        raises(@() nnstab.check_local(A, B, C, net, 1, 'v', 'eig'), 'nnstab:check_local:v') && ...
        raises(@() nnstab.verify(A, B, C, net, 'Direction', -1), 'nnstab:verify:direction'));

    % ---- verify -------------------------------------------------------------
    [Ag, Bg, Cg, netg] = nnstab.example_system('global');
    r = nnstab.verify(Ag, Bg, Cg, netg);
    nfail = nfail + tally('verify: the global example is globally stable', ...
        strcmp(r.status, 'global') && r.stable && isempty(r.local_check) && ...
        nnstab.in_region(r, [5; 1e3; 0]) && ~nnstab.in_region(r, [-1; 0; 0]) && ...
        contains_text(nnstab.report(r), 'Globally exponentially stable'));

    r  = nnstab.verify(A, B, C, net);
    cl = r.local_check;
    past = nnstab.check_local(A, B, C, net, 1.01*cl.ybar);     % just past the edge
    nfail = nfail + tally('verify: the local example finds the same largest region as Python', ...
        strcmp(r.status, 'local') && cl.certified && r.boxes_tried == 17 && ...
        rel(cl.ybar, 1.84765625) && rel(cl.radius, 1.7055217680346539) && ...
        rel(cl.rho, 3.5633309505209767) && rel(cl.v, [1.9285681254405287; 2.0892908066644944; 1]) && ...
        ~past.hurwitz && contains_text(r.note, 'MinDecay') && ...
        contains_text(nnstab.report(r), 'Locally exponentially stable'));

    r = nnstab.verify(A, B, C, net, 'MinDecay', 0.5);
    nfail = nfail + tally('verify with MinDecay = 0.5: a smaller region with a faster rate', ...
        strcmp(r.status, 'local') && r.local_check.decay >= 0.5 && rel(r.local_check.ybar, 1.4404296875) && ...
        rel(r.local_check.radius, 1.3203803027679764) && isempty(r.note));

    r1 = nnstab.verify(A, B, C, net, 'ybar', 1);
    r3 = nnstab.verify(A, B, C, net, 'ybar', 3);
    nfail = nfail + tally('verify on a given box', ...
        strcmp(r1.status, 'local') && r1.boxes_tried == 1 && strcmp(r3.status, 'none') && ~r3.stable && ...
        ~nnstab.in_region(r3, zeros(3, 1)) && contains_text(nnstab.report(r3), 'does not mean'));

    r = nnstab.verify(A + 5*eye(3), B, C, net);
    nfail = nfail + tally('verify: not verified when even small boxes fail', ...
        strcmp(r.status, 'none') && contains_text(r.note, 'very small boxes') && ~r.local_check.hurwitz);

    % pi(y) = -tanh(y) + 0.5 tanh(y) = -0.5 tanh(y), but the global bound is
    % 1.5, too loose for the global test; the local test passes on every box
    r = nnstab.verify([-2 1; 2 -2], [1; 0], [1 0], nnsector.network({[1; 1], [-1 0.5]}));
    nfail = nfail + tally('verify: a note when the local test passes on very large boxes', ...
        ~r.global_check.certified && ~r.global_check.hurwitz && strcmp(r.status, 'local') && ...
        contains_text(r.note, 'very large boxes'));

    % an unstable plant; pi_k(y) = -tanh(2 y_k) - 0.5 tanh(y_k + 0.3 y_other)
    A2 = [-3 2.5 2; 2 -3 2.5; 2.5 2 -4];
    B2 = [1 0; 0 1; 0.5 0.5];
    C2 = [1 0 0.5; 0 1 0.5];
    net2 = nnsector.network({[2 0; 0 2; 1 0.3; 0.3 1], [-1 0 -0.5 0; 0 -1 0 -0.5]});
    r  = nnstab.verify(A2, B2, C2, net2);
    rd = nnstab.verify(A2, B2, C2, net2, 'Direction', [1; 2]);
    ok = strcmp(r.status, 'local') && strcmp(rd.status, 'local') && ~r.global_check.hurwitz && ...
         isequal(size(r.local_check.gamma1), [2 2]) && ...
         rel(r.local_check.ybar, [1.8564453125; 1.8564453125]) && rel(r.local_check.radius, 1.8316911079596425) && ...
         rel(rd.local_check.ybar, [1.3359375; 2.671875]) && rel(rd.local_check.radius, 1.1944858964575569);
    for cc = [r.local_check, rd.local_check]
        Y = C2*diag(cc.rho./cc.v);                       % the outputs at the corners
        ok = ok && all(all(Y <= cc.ybar*(1 + 1e-12))) && abs(max(max(Y./cc.ybar)) - 1) < 1e-12;
    end
    nfail = nfail + tally('two inputs and two outputs, with and without a direction: the same boxes as Python', ok);

    % ---- the results hold in simulation -------------------------------------
    acts = {'tanh', 'relu', 'leakyrelu'};
    ok = true;  nloc = 0;  nglob = 0;  t = 0;
    while (nloc < 8 || nglob < 2) && t < 100       % random loops until enough have results
        t  = t + 1;
        n  = randi([2 4]);
        Ar = 1.5*abs(randn(n)) - (2 + 2*rand)*eye(n);
        Br = abs(randn(n, 1)).*(rand(n, 1) < 0.8);
        Cr = abs(randn(1, n)).*(rand(1, n) < 0.8);
        h  = randi([3 8]);
        Wr = {randn(h, 1), randn(h)/sqrt(h), -abs(randn(1, h))/sqrt(h)};
        netr = nnsector.network(Wr, [], 'Activation', acts{mod(t, 3) + 1});
        r = nnstab.verify(Ar, Br, Cr, netr);
        if strcmp(r.status, 'local')
            nloc = nloc + 1;
            cc = r.local_check;
            X0 = [nnstab.sample_region(cc, 30, t), diag(cc.rho./cc.v)];
            [tt, X] = nnstab.simulate(Ar, Br, Cr, netr, X0, 10, 0.01);
            V = squeeze(sum(X .* reshape(cc.v, 1, n), 2));     % v'*x(t), one column per solution
            ok = ok && min(X(:)) >= -1e-9 && all(V(:) <= cc.rho*(1 + 1e-7)) && ...
                 all(all(V <= V(1, :).*exp(-cc.decay*tt)*(1 + 1e-6) + 1e-12));
        elseif strcmp(r.status, 'global')
            nglob = nglob + 1;
            gg = r.global_check;
            X0 = -10*log(rand(n, 30));
            [tt, X] = nnstab.simulate(Ar, Br, Cr, netr, X0, 10, 0.01);
            S = squeeze(sum(X, 2));                             % sum(x(t)), one column per solution
            ok = ok && min(X(:)) >= -1e-9 && ...
                 all(all(S <= gg.overshoot*exp(-gg.decay*tt).*S(1, :)*(1 + 1e-6) + 1e-12));
        end
    end
    nfail = nfail + tally(sprintf(['%d random loops (%d local, %d global results): the states stay ' ...
                                   '>= 0, in the region, and decay at the certified rate'], t, nloc, nglob), ...
                          ok && nloc >= 8 && nglob >= 2);

    % ---- simulate and sample_region ----------------------------------------
    [tt, X] = nnstab.simulate(-1, 1, 1, nnsector.network({-0.5}), 2, 3, 0.01);
    nfail = nfail + tally('simulate: a linear loop gives x(t) = 2 exp(-1.5 t)', ...
        isequal(size(X), [301 1]) && abs(tt(end) - 3) < 1e-12 && max(abs(X - 2*exp(-1.5*tt))) < 1e-8);

    [~, X] = nnstab.simulate(A, B, C, net, 0.1*ones(3, 5), 1, 0.1);
    nfail = nfail + tally('simulate: many starting states', ...
        isequal(size(X), [11 3 5]) && raises(@() nnstab.simulate(A, B, C, net, ones(2, 1), 1), ...
                                             'nnstab:simulate:size'));

    X = nnstab.sample_region(c, 500, 4);
    nfail = nfail + tally('sample_region: in the region, and up to its edge', ...
        isequal(size(X), [3 500]) && nnstab.in_region(c, X) && max(c.v.'*X) > 0.9*c.rho && ...
        raises(@() nnstab.sample_region(c3, 5), 'nnstab:sample_region:certified'));

    if nfail > 0
        error('nnstab:tests', '%d test(s) failed.', nfail);
    end
    fprintf('All tests passed.\n');
end

% -------------------------------------------------------------------------
function add_nnsector(root)
% Put the MATLAB version of nnsector on the path.
    candidates = {getenv('NNSECTOR_MATLAB'), fullfile(fileparts(fileparts(root)), 'nnsector', 'matlab')};
    for k = 1:numel(candidates)
        d = candidates{k};
        if ~isempty(d) && exist(fullfile(d, '+nnsector'), 'dir')
            addpath(d);
            return
        end
    end
    if isempty(which('nnsector.network'))
        error('nnstab:tests:nnsector', ['The MATLAB version of nnsector was not found. Set the ' ...
              'environment variable NNSECTOR_MATLAB to its folder (the one that contains ' ...
              '+nnsector), or put the nnsector repository next to this one.']);
    end
end

function M = random_metzler_hurwitz(n)
% A random Metzler matrix with spectral abscissa between -2 and -0.1.
    M = abs(randn(n));
    M(1:n+1:end) = 0;
    M = M - (max(real(eig(M))) + 0.1 + 1.9*rand)*eye(n);
end

function f = tally(name, ok)
    if ok
        fprintf('PASS  %s\n', name);
    else
        fprintf('FAIL  %s\n', name);
    end
    f = double(~ok);
end

function t = near(a, b, tol)
    t = all(abs(a(:) - b(:)) <= tol);
end

function t = rel(a, b)
% Equal up to rounding: relative 1e-9 (absolute for numbers below 1).
    t = numel(a) == numel(b) && all(abs(a(:) - b(:)) <= 1e-9*max(1, abs(b(:))));
end

function t = contains_text(s, part)
    t = ~isempty(strfind(s, part));
end

function t = raises(f, id)
    try
        f();
        t = false;
    catch err
        t = strcmp(err.identifier, id);
    end
end
