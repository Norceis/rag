# FINDING
UPDATES_PATTERN = r"(Updates(?: RFC's)?:\s*(.*?)\n)"
OBSOLETES_PATTERN = r"Obsoletes:?+\s*(.*?)\n"
CATEGORY_PATTERN = r"\b(?:Categories?|Category):?\s*(.*?)\n"
ISSN_PATTERN = r"ISSN:?+\s*(.*?)\n"
UPDATED_BY_PATTERN = r"Updated by:?+\s*(.*?)\n"
BCP_PATTERN = r"BCP:?+\s*(.*?)\n"
NIC_PATTERN = r"NIC:?+\s*(.*?)\n"
OBSOLETED_BY_PATTERN = r"Obsoleted by:?+\s*(.*?)\n"
RELATED_RFCS_PATTERN = (
    r"\b(?:Related\s*(?:RFCs|Functional\s*Documents?)?|References)\s*:\s*(.*?)(?:\n|$)"
)

# CLEANING TITLES
COLON_SPACE_PATTERN = r":\s*(.+)"

# CLEANING METADATA
NUMBERS_PATTERN = r"\b\d{1,4}\b"
NIC_CLEAN_PATTERN = r"\b\d{1,10}\b"
ISSN_CLEAN_PATTERN = r"\d{4}-\d{4}"

# CLEANING HEADER
HEADER_PATTERN = (
    r"RFC \d+ .* (?:January|February|March|April|May|June|July|August|September|October|November"
    r"|December) \d{4}"
)
