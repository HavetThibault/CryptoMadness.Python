from file_record_field_type import RecordFieldType


class FileRecordStruct:
    def __init__(self, fields: list[RecordFieldType]):
        self._fields = fields

    def get_struct(self):
        record_struct = []
        for field in self._fields:
            record_struct.append(field.get_tf_type())
        return record_struct
