from pyspark import keyword_only  ## < 2.0 -> pyspark.ml.util.keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
# Available in PySpark >= 2.3.0
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

from pyspark.ml.param.shared import *
# from pyspark.ml.util import keyword_only  # in Spark < 2.0
from pyspark import keyword_only
from pyspark.ml.wrapper import JavaTransformer

class UnicodeNormalizer(JavaTransformer, HasInputCol, HasOutputCol):

    @keyword_only
    def __init__(self, form="NFKD", inputCol=None, outputCol=None):
        super(UnicodeNormalizer, self).__init__()
        self._java_obj = self._new_java_obj(
            "net.zero323.spark.ml.feature.UnicodeNormalizer", self.uid)
        self.form = Param(self, "form",
            "unicode form (one of NFC, NFD, NFKC, NFKD)")
        # kwargs = self.__init__._input_kwargs  # in Spark < 2.0
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, form="NFKD", inputCol=None, outputCol=None):
        # kwargs = self.setParams._input_kwargs  # in Spark < 2.0
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setForm(self, value):
        return self._set(form=value)

    def getForm(self):
        return self.getOrDefault(self.form)