function crop_oct_images(input_dir, output_dir, annot_output_dir, ...
    csv_input_dir ...
)
    %
    % Using a 'threshold' value, it estimates the location of the ILM
    % layer. Then it fits a second degree polynomial to the estimated
    % coordinates. Finally, it uses the min and max curve values and
    % fixed top and bottom margins to crop the OCT image and remove as
    % much background as possible
    %
    % Note that the 'threshold' value used here works with OCT images that
    % have been processed using the 'exponentiation + compensation' method
    % described in:
    %
    % Girard MJ, Strouthidis NG, Ethier CR, Mari JM. Shadow removal and
    % contrast enhancement in optical coherence tomography images of the
    % human optic nerve head. Invest Ophthalmol Vis Sci. 2011;52(10):7738-7748.
    % Published 2011 Sep 29. doi:10.1167/iovs.10-6925
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
    % output_dir: Output path to save the cropped TIFF images.
    %
    % [string]
    % annot_output_dir: Output path to save the input images with the
    % estimates of the ILM (yellow) and the fitted polynomial (green)
    %
    % [boolean]
    % crop_csv: Apply the same cropping to the CSV corresponding to the
    % image
    %
    threshold = 50; % Is computed manually by analysing the intensity of the RPE layer
    crop_top = 30;
    crop_bottom = 200;

    for input_image = dir(input_dir + "/*.tiff")'
        input_image_name = input_image.folder + "/" + input_image.name;
        disp("Processing crop TIFF: " + input_image_name)
        img = imread(input_image_name);
        ilm = NaN(1, size(img, 2));

        for j = 1:size(img, 2)
            ilm(j) = find(img(:,j) >= threshold, 1, 'first');
        end
        % Fit polynomial
        x = 1:length(ilm);
        p = polyfit(x, ilm, 2);
        ilm_fit = round(polyval(p, x));

        img_height = size(img, 1);
        new_top = max(1, max(ilm_fit) - crop_top);
        new_bottom = min(img_height, max(ilm_fit) + crop_bottom);
        if new_top == 1
            disp("WARNING: Top margin reached top of original image")
        end

        if new_bottom == img_height
            disp("WARNING: Bottom margin reached bottom of original image")
        end

        cropped_img = img(new_top : new_bottom, :);
        output_image_name = output_dir + "/" + input_image.name;
        imwrite(cropped_img, output_image_name);

        % Interleave x and ILM
        x_t = x(:).';
        ilm_t = ilm(:).';
        x_ilm_concat = [x_t; ilm_t]; % concatenate them vertically
        x_ilm_interleaved = x_ilm_concat(:);

        % Interleave x and fitted polynomial
        ilm_fit_t = ilm_fit(:).';
        x_ilm_fit_concat = [x_t; ilm_fit_t];
        x_ilm_fit_interleaved = x_ilm_fit_concat(:);

        img = insertShape(img, "line", ...
            [x_ilm_interleaved'; x_ilm_fit_interleaved'], ...
            "Color", ["yellow", "green"] ...
        );

        imwrite(img, annot_output_dir + "/" + input_image.name);

        if exist(csv_input_dir)
            [~, file_name, ~] = fileparts(input_image.name);
            input_csv_file_name = csv_input_dir + "/" + ...
                 file_name + ".csv";
            disp("Processing crop CSV: " + input_csv_file_name);
            mask = readmatrix(input_csv_file_name);
            mask = mask(new_top : new_bottom, :);
            output_csv_file_name = output_dir + "/" + file_name + ".csv";
            writematrix(mask, output_csv_file_name);
        end
    end
end
