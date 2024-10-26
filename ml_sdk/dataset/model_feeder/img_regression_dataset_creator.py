import tensorflow as tf

from img_ds_creator import ImgDsCreator


class ImgRegressionDsCreator(ImgDsCreator):
    def __init__(self, img_dir: str, train_csv_path: str, val_csv_path: str, img_dimensions: list[tuple[int, int]],
                 file_record_struct, repeat_ds: int, batch_size: int, get_img_names, use_cache=False, grey_img=True,
                 get_origin_img_name=None, additional_inputs=0):
        super(ImgRegressionDsCreator, self).__init__(
            img_dir, train_csv_path, val_csv_path, img_dimensions, file_record_struct, repeat_ds,
            batch_size, get_img_names, use_cache, grey_img, get_origin_img_name, additional_inputs)

    def _parse_label_file(self, line_record) -> tuple:
        fields = tf.io.decode_csv(line_record, self._file_record_struct)

        input_dict = self._load_images_tensors(fields[0])

        _, add_inputs_headers, output_headers = self.get_sorted_ds_headers()
        for i in range(1, self._additional_inputs + 1):
            input_dict[add_inputs_headers[i-1]] = fields[i]

        output_dict = {}
        for i, output in enumerate(fields[self._additional_inputs + 1:]):
            output_dict[output_headers[i]] = output

        return input_dict, output_dict
