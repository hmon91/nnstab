function text = report(r)
%REPORT  A short report of a result, in words.
%   nnstab.report(res)           prints it; res from nnstab.verify
%   nnstab.report(g)             g from nnstab.check_global
%   nnstab.report(chk)           chk from nnstab.check_local
%   text = nnstab.report(...)    returns it instead

    if isfield(r, 'status')
        lines = verification_lines(r);
    elseif isfield(r, 'Gamma2')
        lines = global_lines(r);
    else
        lines = local_lines(r);
    end
    out = sprintf('%s\n', lines{:});
    if nargout > 0
        text = out;
    else
        fprintf('%s', out);
    end
end

% -------------------------------------------------------------------------
function lines = verification_lines(r)
    switch r.status
        case 'global'
            g = r.global_check;
            lines = {'Globally exponentially stable: every solution with x0 >= 0 goes to 0.', ...
                     sprintf('  sum(x(t)) <= %.3g exp(-%.3g t) sum(x0)', g.overshoot, g.decay)};
        case 'local'
            c = r.local_check;
            lines = {'Locally exponentially stable (the global test failed).', ...
                     sprintf('  region: {x >= 0 : v''*x <= %.4g},  v = %s', c.rho, fmt(c.v)), ...
                     sprintf('  it holds every x0 >= 0 with sum(x0) <= %.4g', c.radius), ...
                     sprintf('  in it: sum(x(t)) <= %.3g exp(-%.3g t) sum(x0)', c.overshoot, c.decay), ...
                     sprintf('  output box 0 <= y <= %s', fmt(c.ybar))};
        otherwise
            lines = {'Not verified: neither the global nor a local test passed.', ...
                     '  (This does not mean the loop is unstable.)'};
    end
    if ~isempty(r.note)
        lines = [lines, wrap(r.note, '  note: ')];
    end
    lines = [lines, {''}, global_lines(r.global_check)];
    if ~isempty(r.local_check)
        lines = [lines, {''}, local_lines(r.local_check)];
    end
end

function lines = global_lines(g)
    lines = {['Global check: ' certified_word(g.certified)], ...
             sprintf('  global sector slopes  Gamma1 = %s,  Gamma2 = %s', fmt(g.Gamma1), fmt(g.Gamma2)), ...
             sprintf('  A + B Gamma1 C Metzler: %s,  A + B Gamma2 C Hurwitz: %s', yes(g.metzler), yes(g.hurwitz))};
    if g.certified
        lines{end+1} = sprintf('  every x0 >= 0: sum(x(t)) <= %.3g exp(-%.3g t) sum(x0)', ...
                               g.overshoot, g.decay);
    else
        lines = [lines, wrap(g.reason, '  ')];
    end
end

function lines = local_lines(c)
    lines = {sprintf('Local check on 0 <= y <= %s: %s', fmt(c.ybar), certified_word(c.certified)), ...
             sprintf('  local sector slopes  gamma1 = %s,  gamma2 = %s', fmt(c.gamma1), fmt(c.gamma2)), ...
             sprintf('  A + B gamma1 C Metzler: %s,  A + B gamma2 C Hurwitz: %s', yes(c.metzler), yes(c.hurwitz))};
    if c.certified
        lines = [lines, ...
                 {sprintf('  region: {x >= 0 : v''*x <= %.4g},  v = %s', c.rho, fmt(c.v)), ...
                  sprintf('  it holds every x0 >= 0 with sum(x0) <= %.4g', c.radius), ...
                  sprintf('  in it: sum(x(t)) <= %.3g exp(-%.3g t) sum(x0)', c.overshoot, c.decay)}];
    else
        lines = [lines, wrap(c.reason, '  ')];
    end
end

function s = fmt(a)
% A number, a vector or a matrix in short form, such as [1.929, 2.089, 1].
    if isscalar(a)
        s = sprintf('%.4g', a);
    elseif isvector(a)
        s = ['[' strjoin(arrayfun(@(x) sprintf('%.4g', x), a(:).', 'UniformOutput', false), ', ') ']'];
    else
        rows = arrayfun(@(i) fmt(a(i, :)), 1:size(a, 1), 'UniformOutput', false);
        s = ['[' strjoin(rows, ', ') ']'];
    end
end

function lines = wrap(text, first)
% The text indented, in lines of at most 88 characters.
    words = strsplit(text, ' ');
    lines = {};
    line  = [first words{1}];
    for i = 2:numel(words)
        if numel(line) + 1 + numel(words{i}) > 88
            lines{end+1} = line;               %#ok<AGROW>
            line = ['  ' words{i}];
        else
            line = [line ' ' words{i}];        %#ok<AGROW>
        end
    end
    lines{end+1} = line;
end

function s = certified_word(tf)
    if tf, s = 'certified'; else, s = 'not certified'; end
end

function s = yes(tf)
    if tf, s = 'yes'; else, s = 'no'; end
end
