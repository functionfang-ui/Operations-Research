function pos = Dual_Variable(A, price)  
% DUALVARIABLE calculates the dual variables according to the formula cij = ui+vj.
% Input:
% A: The transportation table which has feasible basic variables.
% price: The price table which has the same size as A.
% Output:
% pos: The dual variables that can be used to calculate opportunity cost of
% evey cell.

%Last modified in 2020.4.24 by Hua Siyuan
[m, n] = size(A);
u = [1 0];
v = [];
dual_u = (1:m)'*nan;
dual_u(1) = 0;
dual_v = zeros(1,n)*nan;

while ~isempty(u) || ~isempty(v)
    while ~isempty(u)
        % According to the value of ui, calculate the value of vj
        uidx = u(1);
        value = u(2);
        vidx = ~isnan(A(uidx,:));
        vidx_ = isnan(dual_v);  % In order to avoid endless loop, only choose variables that havn't been calculated.
        vidx = vidx & vidx_;
        dual_v(vidx) = price(uidx,vidx) - value;
        
        % Add newly-added vj to vector v
        vnidx = 1:n;
        vnidx = vnidx(vidx);
        vnidx = vnidx';
        add = [vnidx dual_v(vidx)'];
        v = [v reshape(add',1,[])];
        u(1:2) = [];
    end
    while ~isempty(v)
        vidx = v(1);
        value = v(2);
        uidx = ~isnan(A(:,vidx));
        uidx_ = isnan(dual_u);  % In order to avoid endless loop, only choose variables that havn't been calculated.
        uidx = uidx & uidx_;
        dual_u(uidx) = price(uidx,vidx) - value;
        
        unidx = 1:n;
        unidx = unidx(uidx);
        unidx = unidx';
        add = [unidx dual_u(uidx)];
        u = [u reshape(add',1,[])];
        v(1:2) = [];
    end
end
pos = [dual_u' dual_v];
end
