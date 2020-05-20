from .fit_als import load_als_model
from .fit_lda import load_lda_model, print_topics


def main(spark):

    ldaModel = load_lda_model(spark)
    alsModel = load_als_model()
    print_topics(ldaModel)

