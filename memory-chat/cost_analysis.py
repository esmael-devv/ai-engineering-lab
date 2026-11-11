
import pandas as pd


def calculate_amortized_tokens(
    retrieval_input_tokens,
    retrieval_output_tokens,
    preprocessing_input_tokens,
    preprocessing_output_tokens,
    number_of_queries
):
    if number_of_queries <= 0:
        raise ValueError("number_of_queries must be positive")

    preprocessing_total = (
        preprocessing_input_tokens +
        preprocessing_output_tokens
    )

    retrieval_total = (
        retrieval_input_tokens +
        retrieval_output_tokens
    )

    return {
        "queries": number_of_queries,
        "preprocessing_tokens": preprocessing_total,
        "retrieval_tokens_per_query": retrieval_total,
        "amortized_tokens_per_query": round(
            retrieval_total +
            preprocessing_total / number_of_queries,
            2
        )
    }


def compare_query_volumes():
    results = []

    for queries in [1, 5, 10, 50, 100]:
        result = calculate_amortized_tokens(
            retrieval_input_tokens=158,
            retrieval_output_tokens=46,
            preprocessing_input_tokens=314,
            preprocessing_output_tokens=99,
            number_of_queries=queries
        )

        results.append(result)

    return pd.DataFrame(results)


if __name__ == "__main__":
    print(compare_query_volumes().to_string(index=False))
