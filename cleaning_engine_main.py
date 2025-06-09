import logging
import re
import shutil
from typing import Any, Dict, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from .ce_helpers import *


def _calculate_mixed_fraction_value(
    whole_str: str,
    num_str: str,
    den_str: str,
    logger: logging.Logger,
    context: str,
) -> Tuple[float, str]:
    """
    Calculates the absolute value of a mixed fraction (e.g., "1 1/2").
    Returns (value, error_message). If there's an error, value is np.nan.
    """
    whole = safe_str_to_float(whole_str, logger, f"whole part in {context}")
    numerator = safe_str_to_float(num_str, logger, f"numerator in {context}")
    denominator = safe_str_to_float(den_str, logger, f"denominator in {context}")

    if np.isnan(whole) or np.isnan(numerator) or np.isnan(denominator):
        return np.nan, "Invalid number in mixed fraction components"
    if denominator == 0.0:
        return np.nan, "Division by zero in mixed fraction"
    total = abs(whole) + abs(numerator) / abs(denominator)
    return total, ""


def _calculate_simple_fraction_value(
    num_str: str,
    den_str: str,
    logger: logging.Logger,
    context: str,
) -> Tuple[float, str]:
    """
    Calculates the absolute value of a simple fraction (e.g., "3/4").
    Returns (value, error_message). If there's an error, value is np.nan.
    """
    numerator = safe_str_to_float(num_str, logger, f"numerator in {context}")
    denominator = safe_str_to_float(den_str, logger, f"denominator in {context}")

    if np.isnan(numerator) or np.isnan(denominator):
        return np.nan, "Invalid number in simple fraction components"
    if denominator == 0.0:
        return np.nan, "Division by zero in simple fraction"
    total = abs(numerator) / abs(denominator)
    return total, ""


def _convert_decimal_to_float(
    val_str: str,
    logger: logging.Logger,
    context: str,
) -> Tuple[float, str]:
    """
    Converts a decimal or integer string (possibly containing commas) to absolute float.
    Returns (value, error_message). If there's an error, value is np.nan.
    """
    if pd.isna(val_str) or not isinstance(val_str, str) or not val_str.strip():
        return np.nan, "Empty or invalid decimal/integer string"

    cleaned = val_str.replace(",", "").strip()
    if not cleaned:
        return np.nan, "Empty string after removing commas"
    if cleaned.count(".") > 1:
        return np.nan, "Invalid decimal format: multiple decimal points"

    # Remove leading sign, but keep track of it if needed
    if cleaned[0] in ["+", "-"]:
        cleaned = cleaned[1:].strip()

    if not cleaned:
        return np.nan, "Empty string after stripping sign/commas"

    value = safe_str_to_float(cleaned, logger, f"value part '{cleaned}' in {context}")
    if np.isnan(value):
        return np.nan, "Invalid character(s) in decimal/integer string"
    return abs(value), ""


def _process_row_for_numerical_unit(
    row: pd.Series,
    regex_cols_map: Dict[str, str],
    logger: logging.Logger,
) -> Tuple[float, str]:
    """
    Given a row with extracted regex groups, compute the numeric value (absolute) and
    return (value, error_reason). If no pattern matches, returns (np.nan, reason).
    """
    context = f"attribute_value '{row.get('attribute_value', '')}'"
    val = np.nan
    reason = ""

    # Check for mixed fraction first
    if pd.notna(row[regex_cols_map["mixed_whole"]]):
        val, reason = _calculate_mixed_fraction_value(
            row[regex_cols_map["mixed_whole"]],
            row[regex_cols_map["mixed_num"]],
            row[regex_cols_map["mixed_den"]],
            logger,
            context + " (mixed fraction)",
        )
    # Then check for simple fraction
    elif pd.notna(row[regex_cols_map["simple_num"]]):
        val, reason = _calculate_simple_fraction_value(
            row[regex_cols_map["simple_num"]],
            row[regex_cols_map["simple_den"]],
            logger,
            context + " (simple fraction)",
        )
    # Then check for decimal/integer
    elif pd.notna(row[regex_cols_map["decimal"]]):
        val, reason = _convert_decimal_to_float(
            row[regex_cols_map["decimal"]],
            logger,
            context + " (decimal/integer)",
        )
    else:
        reason = "No recognizable numeric format"
    return val, reason


def clean_numerical_unit(
    df: pd.DataFrame,
    value_column: str = "attribute_value",
    logger: logging.Logger = logging.getLogger(__name__),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cleans numeric values (with or without units) from a DataFrame.
    Returns a tuple of DataFrames: (passed_df, modified_for_error_df, remaining_df).

    - passed_df: rows where numeric extraction succeeded without errors
    - modified_for_error_df: rows where numeric extraction matched but had conversion errors
    - remaining_df: rows where regex did not match at all
    """
    logger.info(f"Starting clean_numerical_unit on {len(df)} rows.")
    if value_column not in df.columns:
        raise KeyError(f"Input DataFrame must contain '{value_column}' column.")

    df_working = df.copy()
    pattern = get_compiled_regex("numeric_with_optional_unit")
    extracted = df_working[value_column].astype(str).str.extract(pattern)
    extracted.columns = [
        "sign",
        "mixed_whole",
        "mixed_num",
        "mixed_den",
        "_unused_simple_whole",
        "simple_num",
        "simple_den",
        "decimal",
        "_unused_dec_int_num",
        "_unused_dec_int_den",
        "unit",
    ]
    extracted.index = df_working.index

    combined = pd.concat([df_working, extracted], axis=1)

    # Rows where none of the numeric groups matched
    no_match_mask = combined["mixed_whole"].isna() & combined["simple_num"].isna() & combined["decimal"].isna()
    remaining_df = combined.loc[no_match_mask, [*df_working.columns]].copy()
    passed_intermediate = combined.loc[~no_match_mask].copy()

    logger.info(f"{len(passed_intermediate)} rows matched regex; {len(remaining_df)} did not.")

    if not passed_intermediate.empty:
        # Determine polarity from sign
        passed_intermediate["polarity1"] = passed_intermediate["sign"].replace({"": pd.NA, np.nan: pd.NA})
        # Assign unit directly from regex group
        passed_intermediate["unit1"] = passed_intermediate["unit"].where(passed_intermediate["unit"].notna(), pd.NA)

        # Map for extraction
        regex_cols_map = {
            "mixed_whole": "mixed_whole",
            "mixed_num": "mixed_num",
            "mixed_den": "mixed_den",
            "simple_num": "simple_num",
            "simple_den": "simple_den",
            "decimal": "decimal",
        }

        # Compute numeric value and error reason
        values_reasons = passed_intermediate.apply(
            lambda row: _process_row_for_numerical_unit(row, regex_cols_map, logger), axis=1
        )
        passed_intermediate[["value1", "mod_reason"]] = pd.DataFrame(
            values_reasons.tolist(), index=passed_intermediate.index
        )

        # Coerce value1 to numeric
        passed_intermediate["value1"] = pd.to_numeric(passed_intermediate["value1"], errors="coerce")

        # Determine data_type based on presence of unit
        has_unit = passed_intermediate["unit1"].notna() & (passed_intermediate["unit1"] != "")
        passed_intermediate["data_type"] = np.where(has_unit, "numerical_with_unit", "numerical_without_unit")

        # Split into passed (no errors) and modified_for_error (errors present)
        error_mask = passed_intermediate["mod_reason"].notna() & (passed_intermediate["mod_reason"] != "")
        mod_df = passed_intermediate.loc[error_mask].copy()
        passed_df = passed_intermediate.loc[~error_mask].copy()
    else:
        passed_df = pd.DataFrame(columns=combined.columns)
        mod_df = pd.DataFrame(columns=combined.columns)

    # Reorder and drop helper columns
    passed_df = check_required_columns(passed_df, logger)
    mod_df = check_required_columns(mod_df, logger)
    remaining_df = check_required_columns(remaining_df, logger)

    logger.info(
        f"Finished clean_numerical_unit: {len(passed_df)} passed, "
        f"{len(mod_df)} had errors, {len(remaining_df)} did not match."
    )
    return passed_df, mod_df, remaining_df


def clean_thread(
    df: pd.DataFrame,
    logger: logging.Logger = logging.getLogger(__name__),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    logger.info(f"Starting clean_thread on {len(df)} rows.")
    if df.empty:
        logger.info(f"Empty df found!")
        cols = list(df.columns)
        return (
            pd.DataFrame(columns=cols),
            pd.DataFrame(columns=cols + ["mod_reason"]),
            pd.DataFrame(columns=cols),
        )

    mask = df["attribute_name"].astype(str).str.contains("thread", case=False, na=False)
    mod_df = df.loc[mask].copy()
    if not mod_df.empty:
        mod_df["mod_reason"] = "spec with thread"
    remaining_df = df.loc[~mask].copy()
    passed_df = pd.DataFrame(columns=df.columns)
    logger.info(
        f"Finished clean_thread: {len(passed_df)} passed, "
        f"{len(mod_df)} had errors, {len(remaining_df)} did not match."
    )
    return passed_df, mod_df, remaining_df


def clean_dimension_values(
    df: pd.DataFrame,
    separators: List[str] | None = None,
    value_column: str = "attribute_value",
    logger: logging.Logger = logging.getLogger(__name__),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if separators is None:
        separators = [" x ", " X "]
    logger.info(f"Starting clean_dimension_values with separator: {separators} on {len(df)} rows.")

    if df.empty:
        logger.info(f"Empty df found!")
        empty_cols = list(df.columns) + ["key_value"]
        return (
            pd.DataFrame(columns=empty_cols),
            pd.DataFrame(columns=empty_cols + ["mod_reason"]),
            pd.DataFrame(columns=empty_cols),
        )

    split_pattern = "|".join(re.escape(sep) for sep in separators)

    work = df.copy()
    work["_orig_attr"] = work[value_column]
    work["_has_sep"] = work[value_column].astype(str).str.contains(split_pattern, regex=True)
    work["_orig_index"] = work.index
    work["_parts"] = work[value_column].astype(str).str.split(split_pattern, regex=True)

    exploded = work.explode("_parts").copy()
    exploded[value_column] = exploded["_parts"].str.strip()
    exploded["key_value"] = exploded.apply(lambda r: r["_orig_index"] if r["_has_sep"] else pd.NA, axis=1)

    # drop helper cols *before* handing to numeric cleaner
    exploded.drop(columns=["_parts", "_has_sep", "_orig_index"], inplace=True)

    passed_df, mod_df, remaining_df = clean_numerical_unit(exploded, value_column=value_column)

    # Restore original attribute_value ------------------------------------------------
    for _df in (passed_df, mod_df, remaining_df):
        if not _df.empty and "_orig_attr" in _df.columns:
            _df[value_column] = _df["_orig_attr"]
            _df.drop(columns=["_orig_attr"], inplace=True)

    logger.info(
        f"Finished clean_dimension_values: {len(passed_df)} passed, "
        f"{len(mod_df)} had errors, {len(remaining_df)} did not match."
    )

    return passed_df, mod_df, remaining_df


def clean_tolerance_values(
    df: pd.DataFrame,
    unit_df: pd.DataFrame,
    value_column: str = "attribute_value",
    logger: logging.Logger = logging.getLogger(__name__),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cleans tolerance values like '+/- 0.05"' into a standardized range format.
    """
    logger.info(f"Starting clean_tolerance_values on {len(df)} rows.")
    if df.empty:
        logger.warning("Input DataFrame for tolerance cleaning is empty.")
        return df, pd.DataFrame(columns=df.columns), df

    pattern = get_compiled_regex("tolerance")

    # Ensure value column is string for regex matching
    df_work = df.copy()
    df_work[value_column] = df_work[value_column].astype(str)

    # Rows that do not match the tolerance pattern
    match_mask = df_work[value_column].str.contains(pattern, na=False)
    remaining_df = df_work[~match_mask].copy()

    # Rows that match the pattern
    to_process_df = df_work[match_mask].copy()

    if to_process_df.empty:
        logger.info("No rows matched the tolerance pattern.")
        return pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns), remaining_df

    logger.info(f"{len(to_process_df)} rows matched the tolerance pattern.")

    # Extract value and unit from matching rows
    extracted_parts = to_process_df[value_column].str.extract(pattern)
    to_process_df["value1"] = pd.to_numeric(extracted_parts[0], errors="coerce")
    to_process_df["unit1"] = extracted_parts[1].str.strip()

    # Create the range representation
    to_process_df["value2"] = to_process_df["value1"]
    to_process_df["unit2"] = to_process_df["unit1"]
    to_process_df["polarity1"] = "-"
    to_process_df["polarity2"] = "+"
    to_process_df["data_type"] = "range1"

    # Clean up the processed range values
    passed_df, mod_df = cleanup_range(to_process_df, unit_df, logger)

    logger.info(
        f"Finished clean_tolerance_values: {len(passed_df)} passed, "
        f"{len(mod_df)} had errors, {len(remaining_df)} did not match."
    )

    return passed_df, mod_df, remaining_df


def clean_range_with_to_and_plus_minus(
    df: pd.DataFrame,
    unit_df: pd.DataFrame,
    value_column: str = "attribute_value",
    logger: logging.Logger = logging.getLogger(__name__),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits values containing a range using ' to ' (e.g., "100 to 500 km").
    Calls `clean_dimension_values` with separator [' to '], then pivots the two exploded rows
    per key_value into a single row with columns:
      - polarity1, value1, unit1 (from the first part)
      - polarity2, value2, unit2 (from the second part)
      - data_type set to literal 'range1'

    Returns:
      - range_df: DataFrame of successfully parsed ranges (one row per key_value)
      - mod_df: DataFrame of any rows that had conversion errors from numeric cleaning
      - remaining_df: DataFrame of any rows that could not be parsed as dimensions or numbers
    """
    logger.info(f"Starting clean_range_with_to_and_plus_minus on {len(df)} rows.")

    if df.empty:
        logger.info(f"Empty df found!")
        empty_cols = list(df.columns) + ["key_value"]
        return (
            pd.DataFrame(columns=empty_cols),
            pd.DataFrame(columns=empty_cols + ["mod_reason"]),
            pd.DataFrame(columns=empty_cols),
        )

    passed_df, mod_df, remaining_df = clean_dimension_values(
        df, separators=[" to ", "/", " / "], value_column=value_column, logger=logger
    )

    # If no passed rows, return empty range_df plus mod and remaining as-is
    if passed_df.empty:
        empty_cols = list(df.columns) + [
            "polarity1",
            "value1",
            "unit1",
            "polarity2",
            "value2",
            "unit2",
            "data_type",
            "key_value",
        ]
        return (pd.DataFrame(columns=empty_cols), mod_df, remaining_df)

    range_rows: list = []
    for key, group in passed_df.groupby("key_value", sort=False):
        grp_sorted = group.sort_index()
        first = grp_sorted.iloc[0]
        if len(grp_sorted) > 1:
            second = grp_sorted.iloc[1]
            combined = first.copy()
            combined["polarity2"] = second["polarity1"]
            combined["value2"] = second["value1"]
            combined["unit2"] = second["unit1"]
        else:
            combined = first.copy()
            combined["polarity2"] = pd.NA
            combined["value2"] = pd.NA
            combined["unit2"] = pd.NA
        combined["data_type"] = "range1"
        range_rows.append(combined)

    # Assemble DataFrame; reset index to clean up
    range_df = pd.DataFrame(range_rows).reset_index(drop=True)

    range_clean_pass, range_clean_mod = cleanup_range(range_df, unit_df)
    mod_df = pd.concat([mod_df, range_clean_mod], ignore_index=True)

    logger.info(
        f"Finished clean_range_with_to_and_plus_minus: "
        f"{len(range_clean_pass)} passed, {len(range_clean_mod)} had errors, {len(remaining_df)} did not match."
    )

    return range_clean_pass, mod_df, remaining_df


def clean_varchar_and_categorical(
    df: pd.DataFrame,
    value_column: str = "attribute_value",
    logger: logging.Logger = logging.getLogger(__name__),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cleans non-numeric, varchar, and categorical values.

    - Rows with digits are moved to `remaining_df`.
    - Rows with commas are split, exploded, and classified as 'categorical2'.
    - Rows without commas are classified as 'categorical' (<=3 words) or 'varchar' (>3 words).

    Returns a tuple of DataFrames: (passed_df, modified_for_error_df, remaining_df).
    """
    logger.info(f"Starting clean_varchar_and_categorical on {len(df)} rows.")

    if df.empty:
        logger.warning("Input DataFrame is empty.")
        cols = list(df.columns)
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)

    # Ensure value column is string type for reliable processing
    df_work = df.copy()
    df_work[value_column] = df_work[value_column].astype(str)

    # Filter out rows containing any digit
    has_digit_mask = df_work[value_column].str.contains(r"\d", na=False)
    remaining_df = df_work[has_digit_mask].copy()
    to_process_df = df_work[~has_digit_mask].copy()

    logger.info(f"{len(remaining_df)} rows contain digits and are moved to remaining.")
    logger.info(f"{len(to_process_df)} rows to be processed for categorical/varchar.")

    passed_frames = []

    # Process rows with commas
    has_comma_mask = to_process_df[value_column].str.contains(",", na=False)
    with_comma_df = to_process_df[has_comma_mask].copy()
    without_comma_df = to_process_df[~has_comma_mask].copy()

    if not with_comma_df.empty:
        logger.info(f"Processing {len(with_comma_df)} rows with commas.")
        with_comma_df["_orig_attr"] = with_comma_df[value_column]
        with_comma_df["_orig_index"] = with_comma_df.index

        # Explode into multiple rows
        with_comma_df["value1"] = with_comma_df[value_column].str.split(",")
        exploded_comma = with_comma_df.explode("value1")
        exploded_comma["value1"] = exploded_comma["value1"].str.strip()

        # Set key_value and data_type
        exploded_comma["key_value"] = exploded_comma["_orig_index"]
        exploded_comma["data_type"] = "categorical2"
        exploded_comma[value_column] = exploded_comma["_orig_attr"]
        exploded_comma.drop(columns=["_orig_attr", "_orig_index"], inplace=True)

        passed_frames.append(exploded_comma)
        logger.info(f"Exploded {len(with_comma_df)} rows into {len(exploded_comma)} categorical2 rows.")

    # Process rows without commas
    if not without_comma_df.empty:
        logger.info(f"Processing {len(without_comma_df)} rows without commas.")
        word_counts = without_comma_df[value_column].str.split().str.len()

        # Categorical: 3 or less words
        categorical_mask = word_counts <= 3
        categorical_df = without_comma_df[categorical_mask].copy()
        if not categorical_df.empty:
            categorical_df["data_type"] = "categorical"
            categorical_df["value1"] = categorical_df[value_column]
            passed_frames.append(categorical_df)
            logger.info(f"Classified {len(categorical_df)} rows as 'categorical'.")

        # Varchar: 4 or more words
        varchar_mask = word_counts >= 4
        varchar_df = without_comma_df[varchar_mask].copy()
        if not varchar_df.empty:
            varchar_df["data_type"] = "varchar"
            varchar_df["value1"] = varchar_df[value_column]
            passed_frames.append(varchar_df)
            logger.info(f"Classified {len(varchar_df)} rows as 'varchar'.")

    # Combine all passed rows
    if passed_frames:
        passed_df = pd.concat(passed_frames, ignore_index=True)
    else:
        passed_df = pd.DataFrame(columns=df.columns)

    # For this function, we don't have explicit error modifications, so mod_df is empty
    mod_df = pd.DataFrame(columns=df.columns)

    # Ensure all DataFrames have standard columns before returning
    passed_df = check_required_columns(passed_df, logger)
    mod_df = check_required_columns(mod_df, logger)
    remaining_df = check_required_columns(remaining_df, logger)

    logger.info(
        f"Finished clean_varchar_and_categorical: {len(passed_df)} passed, "
        f"{len(mod_df)} had errors, {len(remaining_df)} remaining."
    )
    return passed_df, mod_df, remaining_df


def run_cleanup_pipeline(
    raw_df: pd.DataFrame,
    unit_df: pd.DataFrame,
    base_dir: str = ".",
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    """Run the full CE cleanup pipeline and materialise intermediate outputs.

    The sequence is:
      1. ce_start_cleanup
      2. clean_numerical_unit
      3. clean_thread
      4. clean_dimension_values
      5. clean_tolerance_values
      6. clean_range_with_to_and_plus_minus
      7. clean_varchar_and_categorical

    *passed* frames go to ``<base_dir>/passed``; *mod* frames (including the final
    *remain*) to ``<base_dir>/mod``.
    """

    if Path(base_dir).exists():
        shutil.rmtree(base_dir)
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting cleanup pipeline with base directory: {base_dir}")
    logger.info(f"Input DataFrame has {len(raw_df)} rows and {len(raw_df.columns)} columns.")

    # 0 – start‑of‑pipe cleanup
    df_after_start = ce_start_cleanup(raw_df)

    logger.info(f"After initial cleanup, {len(df_after_start)} rows remain.")
    remain = df_after_start

    # 1 – numerical units -------------------------------------------------------
    passed, mod, remain = clean_numerical_unit(remain)
    save_dfs("clean_numerical_unit", passed, mod, base_dir)

    # 2 – thread spec filter ----------------------------------------------------
    passed, mod, remain = clean_thread(remain)
    save_dfs("clean_thread", passed, mod, base_dir)

    # 3 – dimension (X × Y) values --------------------------------------------
    passed, mod, remain = clean_dimension_values(remain)
    save_dfs("clean_dimension_values", passed, mod, base_dir)

    # 4 - tolerance values (+/-) ---------------------------------------------
    passed, mod, remain = clean_tolerance_values(remain, unit_df)
    save_dfs("clean_tolerance_values", passed, mod, base_dir)

    # 5 – ranges ("100 to 200") ------------------------------------------------
    passed, mod, remain = clean_range_with_to_and_plus_minus(remain, unit_df)
    save_dfs("clean_range_with_to_and_plus_minus", passed, mod, base_dir)

    # 6 – varchar and categorical ----------------------------------------------
    passed, mod, remain = clean_varchar_and_categorical(remain)
    save_dfs("clean_varchar_and_categorical", passed, mod, base_dir)

    # 7 – whatever is *still* left becomes final remain and is saved as mod -----
    if not remain.empty:
        remain["mod_reason"] = "final_unprocessed"
        remain_path = os.path.join(base_dir, "mod", "final_remain.csv")
        remain.to_csv(remain_path, index=False)
