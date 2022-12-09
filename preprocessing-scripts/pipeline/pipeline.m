function pipeline( ... 
    input_dir, ce_output_dir, trimmed_ce_output_dir, output_dir, ...
    annot_output_dir, resized_width, resized_height ...
)
    %
    % This function takes a input directory containing OCT images in .tiff
    % format and their corresponding masks in CSV format and performs the
    % following operations:
    % 1. Contrast Enhancement
    % 2. Cropping to remove background
    % 3. Resizing
    %
    % Note: This function requires the MATLAB "Computer Vision Toolbox"
    %
    % Input
    % -----
    % [string]
    % input_dir: Path to the input directory. The TIFF files found will be
    % processed.
    %
    % [string]
    % ce_output_dir: Path to save the contrast-enhanced images.
    %
    % [string]
    % trimmed_ce_output_dir: Path to save the cropped contrast-enhanced
    % images.
    %
    % [string]
    % output_dir: Output path to save the resized cropped contrast-enhanced
    % TIFF images and CSVs.
    %
    % [string]
    % annot_output_dir: Output path to save the input images with the
    % estimates of the ILM (yellow) and the fitted polynomial (green)
    %
    % [int]
    % resized_width: Width to which the images should be resized to.
    %
    % [int]
    % resized_height: Height to which the images should be resized to.
    %
    % Sample usage:
    %
    % >> pipeline("test-in", "ce-test-out", "trimmed-ce-test-out", ...
    %        "test-out", "test-annot", 384, 480)
    %
    addpath("oct-image-contrast-enhancement/");
    addpath("trimming/");

    % Contrast Enhancement
    enhance_images_contrast(input_dir, ce_output_dir);

    % Cropping to remove background
    crop_oct_images(ce_output_dir, trimmed_ce_output_dir, ...
        annot_output_dir, input_dir ...
    );

    % Resizing
    for trimmed_input_image = dir(trimmed_ce_output_dir + "/*.tiff")'
        input_image_name = trimmed_input_image.folder + "/" + ...
            trimmed_input_image.name;
        disp("Processing Image resize: " + input_image_name)
        img = imread(input_image_name);
        resized_img = imresize(img, [resized_height, resized_width]);
        output_image_name = output_dir + "/" + trimmed_input_image.name;
        imwrite(resized_img, output_image_name);
            
        [~, mask_file_name, ~] = fileparts(trimmed_input_image.name);
        mask_image_name = trimmed_input_image.folder + "/" + ...
        mask_file_name + ".csv";
        disp("Processing Mask resize: " + mask_image_name);
        mask = readmatrix(mask_image_name);
        resized_mask = imresize(mask, [resized_height, resized_width], ...
            "nearest" ...
        );
        output_mask_name = output_dir + "/" + mask_file_name + ".csv";
        writematrix(resized_mask, output_mask_name);
    end
end
