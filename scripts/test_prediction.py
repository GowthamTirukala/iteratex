"""Test script to verify model registration and prediction."""

import sys

import requests


def test_prediction():
    """Test the prediction endpoint with sample data."""
    # Sample data that matches the model's expected features
    sample_data = {
        "length_url": 100,
        "length_hostname": 20,
        "ip": 0,
        "nb_dots": 2,
        "nb_hyphens": 1,
        "nb_at": 0,
        "nb_qm": 1,
        "nb_and": 2,
        "nb_or": 0,
        "nb_eq": 1,
        "nb_underscore": 0,
        "nb_tilde": 0,
        "nb_percent": 0,
        "nb_slash": 5,
        "nb_star": 0,
        "nb_colon": 1,
        "nb_comma": 0,
        "nb_semicolumn": 0,
        "nb_dollar": 0,
        "nb_space": 0,
        "nb_www": 1,
        "nb_com": 1,
        "nb_dslash": 1,
        "http_in_path": 0,
        "https_token": 1,
        "ratio_digits_url": 0.1,
        "ratio_digits_host": 0.0,
        "punycode": 0,
        "port": 1,
        "tld_in_path": 0,
        "tld_in_subdomain": 0,
        "abnormal_subdomain": 0,
        "nb_subdomains": 2,
        "prefix_suffix": 0,
        "random_domain": 0,
        "shortening_service": 0,
        "path_extension": 1,
        "nb_redirection": 1,
        "nb_external_redirection": 0,
        "length_words_raw": 30,
        "char_repeat": 0.1,
        "shortest_words_raw": 3,
        "shortest_word_host": 4,
        "shortest_word_path": 3,
        "longest_words_raw": 12,
        "longest_word_host": 8,
        "longest_word_path": 12,
        "avg_words_raw": 6.5,
        "avg_word_host": 5.0,
        "avg_word_path": 6.0,
        "phish_hints": 0,
        "domain_in_brand": 1,
        "brand_in_subdomain": 0,
        "brand_in_path": 1,
        "suspecious_tld": 0,
        "statistical_report": 0.8,
    }

    url = "http://localhost:8000/predict"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=sample_data, headers=headers)
        response.raise_for_status()
        result = response.json()

        print("✅ Prediction successful!")
        print(f"Model version: {result['model_version']}")
        print(f"Prediction: {result['prediction']}")
        return 0
    except requests.exceptions.RequestException as e:
        print(f"❌ Error making prediction: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return 1


if __name__ == "__main__":
    print("🚀 Testing model prediction...")
    sys.exit(test_prediction())
