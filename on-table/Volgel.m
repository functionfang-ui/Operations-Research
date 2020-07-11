function A = Vogel_method(A,price,prod,sell,m,n)
penalty_row = 1:n;
penalty_line = (1:m)';
indexr = 1:m;
indexl = 1:n;
dual = zeros(m+n-1,2);  %Used to determine the position to add zero basic variable.
num = 1;        %Used to indicate the number of iteration
while ~isempty(prod)
    temp = sort(price,2);
    [m, n] = size(price);
    if n >= 2   %Calulate the difference between smallest and next highest price.
        penalty_line = temp(:,2) - temp(:,1);
    else        %The temp is used as the penalty_line 
        penalty_line = temp;
    end
    temp = sort(price,1);
    if m >= 2   %Calulate the difference between smallest and next highest price.
        penalty_row = temp(2,:) - temp(1,:);
    else        %The temp is used as the penalty_row
        penalty_row = temp;
    end
    
    %Get the unit to fill number.
    [maxr,maxrarg] = max(penalty_row);
    [maxl,maxlarg] = max(penalty_line);
    unitr = 0;
    unitl = 0;
    if maxl >= maxr
        unitr = maxlarg;
        [~, unitl] = min(price(unitr,:));
    else
        unitl = maxrarg;
        [~, unitr] = min(price(:,unitl));
    end
    
    %Modify the prod,sell,dual and price
    dual(num,:) = [indexr(unitr), indexl(unitl)];
    min_ = min(prod(unitr), sell(unitl));
    prod(unitr) = prod(unitr) - min_;
    sell(unitl) = sell(unitl) - min_;
    A(indexr(unitr), indexl(unitl)) = min_;
    if prod(unitr) == 0
        prod(unitr) = [];
        price(unitr,:) = [];
        indexr(unitr) = [];
    end
    if sell(unitl) == 0
        sell(unitl) = [];
        price(:,unitl) = [];
        indexl(unitl) = [];
    end
    num = num + 1;
end
[m,n] = size(A);
if num <= m+n-1     %Need add zero basic variable
    dual(num:end,:) = [];   % num-1 formulas in total. The last row doesn't hold a formula.
    addu = [];
    addv = [];
    while ~isempty(dual)
        addv = dual(1,2);   % Used to add a formula for calculating dual variables.
        u = [dual(1,1), 0];
        while ~isempty(u)%sum(sum(~isnan(dual))) > 2    %In this situation, one formulation will be left unsolved, which means two elements are left in the dual
            rindx = dual(:,mod(u(2),2)+1)==u(1);
            duall = dual(rindx, mod(u(2)+1,2)+1);
            duall(:,end+1) = u(2)+1;
            u = [u reshape(duall',1,[])];
            u(1:2) = [];
            dual(rindx,:) = [];
        end
        ridx = ~isnan(dual(:,1));
        left = dual(ridx,:);    %The left formulas.
        if isempty(left)
            break;
        end
        A(left(1,1), addv) = 0;
    end
end
end
