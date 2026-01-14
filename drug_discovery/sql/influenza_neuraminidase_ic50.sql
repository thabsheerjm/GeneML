SELECT
    act.standard_value,
    act.standard_units,
    act.standard_relation,
    cs.canonical_smiles
FROM target_dictionary td
JOIN assays a
    ON td.tid = a.tid
JOIN activities act
    ON a.assay_id = act.assay_id
JOIN compound_structures cs
    ON act.molregno = cs.molregno
WHERE td.chembl_id = 'CHEMBL6135'
    AND act.standard_type = 'IC50'
    AND act.standard_value IS NOT NULL;
