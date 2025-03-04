function exp_comp(input_path, output_path)
    %
    % Applies 'exponentiation' and 'compensation' to the TIFF file

    % Methods based on:
    % Girard MJ, Strouthidis NG, Ethier CR, Mari JM. Shadow removal and
    % contrast enhancement in optical coherence tomography images of the human
    % optic nerve head. Invest Ophthalmol Vis Sci. 2011;52(10):7738-7748.
    % Published 2011 Sep 29. doi:10.1167/iovs.10-6925
    %
    % Input
    % -----
    % [string]
    % input_path: Path to the input TIFF image.
    %
    % [string]
    % output_path: Path to save the processed TIFF image.
    %
    t = Tiff(input_path,'r');
    I = read(t);
    I = double(I);
    I = (I/255).^4;
    I = (I.^2)./ (flipud(cumtrapz(flipud(I.^2)))); % L (Equation A8 - Appendix)
    I = nthroot(I, 4)*255;
    I = uint8(I);
    imwrite(I, output_path)
end
