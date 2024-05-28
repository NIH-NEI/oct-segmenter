function comp_only(path)
    %
    % Applies 'compensation' to the TIFF file

    % Methods based on:
    % Girard MJ, Strouthidis NG, Ethier CR, Mari JM. Shadow removal and
    % contrast enhancement in optical coherence tomography images of the human
    % optic nerve head. Invest Ophthalmol Vis Sci. 2011;52(10):7738-7748.
    % Published 2011 Sep 29. doi:10.1167/iovs.10-6925
    %
    % Input
    % -----
    % [string]
    % path:  Path to the input TIFF image.
    %
    t = Tiff(path,'r');
    I = read(t);
    I = double(I);
    I = (I/255).^4;
    I = I./ (flipud (cumtrapz(flipud(I))));  % J (Equation A6 - Appendix)
    I = nthroot(I, 4)*255;
    I = uint8(I);
    imwrite(I, path(1:end-5) + "_comp_only.tiff")
end
