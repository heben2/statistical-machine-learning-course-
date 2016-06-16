function resHandle = betterPlots(a)
    allaxes=findall(a,'Type','axes');
    alllines=findall(a,'Type','line');
    alltext=findall(a,'Type','text');

    set(allaxes,'FontName','Arial','FontWeight','Bold','LineWidth',2, 'FontSize',14);
    set(alllines,'Linewidth',1);
    set(alltext,'FontName','Arial','FontWeight','Bold','FontSize',14);
    grid on;
    resHandle = a;
end