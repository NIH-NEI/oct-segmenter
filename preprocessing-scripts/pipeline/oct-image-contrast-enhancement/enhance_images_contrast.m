function enhance_images_contrast(input_dir, output_dir)
    %
    % Applies 'exponentiation' and 'compensation' to the TIFF files found
    % in the input directory

    % Methods based on:
    % Girard MJ, Strouthidis NG, Ethier CR, Mari JM. Shadow removal and
    % contrast enhancement in optical coherence tomography images of the human
    % optic nerve head. Invest Ophthalmol Vis Sci. 2011;52(10):7738-7748.
    % Published 2011 Sep 29. doi:10.1167/iovs.10-6925
    %
    % Input
    % -----
    % [string]
    % input_dir: Path to the input directory. The TIFF files found will be
    % processed.
    %
    % [string]
    % output_dir: Output path to save the processed TIFF images.
    %
    for input_image = dir(input_dir + "/*.tiff")'
        disp("Processing CE: " + input_image.folder + "/" + ...
            input_image.name)
        exp_comp( ...
            input_image.folder + "/" + input_image.name, ...
            output_dir + "/" + input_image.name ...
        )
    end
end
