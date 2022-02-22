function imout = paste_mask(orig_img, orig_img_pos_to_parseat, obj_to_parse, ...
    obj_to_parse_start_pos, mask_obj)
% This function pastes obj_to_parse onto orig_img and returns the resulting image.
% orig_img_pos_to_parseat:   Target position in orig_img, defined by top-left point where to
% parse the object
% obj_to_parse_start_pos:   Source position in obj_to_parse, defined by top-left point starting
% from which to parse the object
% mask_obj:     Opacity mask for obj_to_parse. For 0-values, orig_img is unchanged
% Pixels outside the valid range are removed.
% You may use this function in your code.

imout = orig_img;
szm = size(mask_obj);
szi = size(orig_img);
szo = size(obj_to_parse);

% object
o_cols = obj_to_parse_start_pos(1):(obj_to_parse_start_pos(1)+szm(2)-1);
o_rows = obj_to_parse_start_pos(2):(obj_to_parse_start_pos(2)+szm(1)-1);
o_cols_ok = and(o_cols>0, o_cols<=szo(2));
o_rows_ok = and(o_rows>0, o_rows<=szo(1));

% background/output
i_cols = orig_img_pos_to_parseat(1):(orig_img_pos_to_parseat(1)+szm(2)-1);
i_rows = orig_img_pos_to_parseat(2):(orig_img_pos_to_parseat(2)+szm(1)-1);
i_cols_ok = and(i_cols>0, i_cols<=szi(2));
i_rows_ok = and(i_rows>0, i_rows<=szi(1));

cols_ok = and(i_cols_ok, o_cols_ok);
rows_ok = and(i_rows_ok, o_rows_ok);
imout(i_rows(rows_ok), i_cols(cols_ok)) = ...
    (1-mask_obj(rows_ok, cols_ok)) .* orig_img(i_rows(rows_ok), i_cols(cols_ok)) + ...
    mask_obj(rows_ok, cols_ok) .* obj_to_parse(o_rows(rows_ok), o_cols(cols_ok));