function r = graph_largest_component_ratio(A)
%GRAPH_LARGEST_COMPONENT_RATIO Largest component size ratio in an undirected graph.
%   R = GRAPH_LARGEST_COMPONENT_RATIO(A) takes a logical adjacency matrix A
%   (assumed symmetric) and returns the fraction of nodes in the largest
%   connected component using BFS/union-find (no toolboxes required).

N = size(A,1);
visited = false(N,1);
best = 0;

for i = 1:N
    if visited(i)
        continue;
    end
    % BFS
    q = i;
    visited(i) = true;
    count = 0;
    while ~isempty(q)
        v = q(1); q(1) = [];
        count = count + 1;
        neighbors = find(A(v,:));
        for nb = neighbors
            if ~visited(nb)
                visited(nb) = true;
                q(end+1) = nb; %#ok<AGROW>
            end
        end
    end
    if count > best
        best = count;
    end
end

r = best / max(1,N);
end
