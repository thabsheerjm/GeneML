SELECT
    chembl_id,
    pref_name,
    organism
FROM target_dictionary
WHERE organism LIKE '%Influenza%'
    AND organism NOT LIKE '%Haemophilus%';
