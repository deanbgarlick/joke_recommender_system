from modelling.fit_als import load_als_model
from modelling.fit_lda import load_lda_model


def main():

    ldaModel = load_lda_model()
    alsModel = load_als_model()


if __name__ == "__main__":
    main()
