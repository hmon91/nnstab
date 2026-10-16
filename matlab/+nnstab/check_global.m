function g = check_global(A, B, C, net)
%CHECK_GLOBAL  Global test: the positive Aizerman condition with the global sector bound.
%   g = nnstab.check_global(A, B, C, net)
%
%   Gamma2 = c^q |W_{q+1}| ... |W_1| (nnsector.global_bound) and
%   Gamma1 = -Gamma2. Certified if A + B*Gamma1*C is Metzler and
%   A + B*Gamma2*C is Hurwitz; then the origin is globally exponentially
%   stable for all x0 >= 0:
%       sum(x(t)) <= g.overshoot * exp(-g.decay*t) * sum(x0).
%
%   Fields: Gamma1, Gamma2, M1 = A + B*Gamma1*C, M2 = A + B*Gamma2*C,
%   metzler, hurwitz, certified, decay, v (with v'*M2 <= -decay*v' and
%   smallest entry 1), overshoot = max(v)/min(v), and reason (why the test
%   failed, '' if it passed). nnstab.report(g) prints a report.

    [A, B, C, net] = nnstab.check_system(A, B, C, net);
    gb = nnsector.global_bound(net);
    G2 = gb.Gamma2;
    G1 = -G2;
    M1 = A + B*G1*C;
    M2 = A + B*G2*C;
    g = struct('Gamma1', G1, 'Gamma2', G2, 'M1', M1, 'M2', M2, ...
               'metzler', nnstab.is_metzler(M1), 'hurwitz', nnstab.is_hurwitz(M2), ...
               'certified', false, 'decay', 0, 'v', [], 'overshoot', [], 'reason', '');
    if g.metzler && g.hurwitz
        [g.v, g.decay] = nnstab.copositive_vector(M2);
        g.certified = true;
        g.overshoot = max(g.v)/min(g.v);
    elseif ~g.metzler
        g.reason = ['A + B Gamma1 C is not Metzler: the global lower slope -Gamma2 is too ' ...
                    'negative to keep the states nonnegative.'];
    else
        g.reason = 'A + B Gamma2 C is not Hurwitz: the global upper slope Gamma2 is too large.';
    end
end
