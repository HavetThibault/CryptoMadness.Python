import tensorflow as tf

from img_ds_creator import ImgDsCreator


class ImgClassificationDsCreator(ImgDsCreator):
    def __init__(self, img_dir: str, train_csv_path: str, val_csv_path: str, img_dimensions: list[tuple[int, int]],
                 file_record_struct, repeat_ds: int, batch_size: int, get_img_names, categories: list[int],
                 use_cache=False, grey_img=True, get_origin_img_name=None, additional_inputs=0):
        super(ImgClassificationDsCreator, self).__init__(
            img_dir, train_csv_path, val_csv_path, img_dimensions, file_record_struct, repeat_ds,
            batch_size, get_img_names, use_cache, grey_img, get_origin_img_name, additional_inputs)
        self._categories = categories

    def _convert_to_categories(self, fields):
        categories_fields = []
        i = 0
        for category in self._categories:
            category_tensor = []
            for k in range(category):
                category_tensor.append(fields[i])
                i += 1
            categories_fields.append(tf.convert_to_tensor(category_tensor))
        if len(categories_fields) == 1:
            return categories_fields[0]
        return tuple(categories_fields)

    def _parse_label_file(self, line_record) -> tuple:
        fields = tf.io.decode_csv(line_record, self._file_record_struct)
        img_nbr = self.get_img_nbr()

        input_dict = self._load_images_tensors(fields[0])

        _, add_inputs_headers, output_headers = self.get_sorted_ds_headers()
        for i in range(1, self._additional_inputs+1):
            input_dict[add_inputs_headers[i-1]] = fields[i]

        output_dict = {}
        for i, output in enumerate(self._convert_to_categories(fields[self._additional_inputs + img_nbr:])):
            output_dict[output_headers[i]] = output

        return input_dict, output_dict
