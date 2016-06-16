function result = rotateMatrix(M, d)
    r = d * pi/180;
    R = [cos(r) -sin(r); sin(r) cos(r)];
    result = R^(-1) * M * R;
end