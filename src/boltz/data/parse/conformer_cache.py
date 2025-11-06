"""Conformer caching functionality for SMILES ligands."""

import pickle
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem

from boltz.data.parse.schema import compute_3d_conformer


def get_mol_with_conformer(
    smiles: str,
    affinity: bool,
    cache_dir: Optional[Path] = None,
) -> Chem.Mol:
    """Get a molecule with 3D conformer, using cache if available.

    This function caches conformers based on the InChIKey of the molecule.
    If affinity is True, conformers are stored in a separate 'affinity' subdirectory.

    Parameters
    ----------
    smiles : str
        The SMILES string of the molecule
    affinity : bool
        Whether this molecule is used for affinity prediction
    cache_dir : Path, optional
        The cache directory. If None, no caching is performed.

    Returns
    -------
    Chem.Mol
        The molecule with 3D conformer

    Raises
    ------
    ValueError
        If conformer generation fails
    """
    # Create mol with hydrogens
    mol = AllChem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)

    # If no cache directory, just compute conformer
    if cache_dir is None:
        success = compute_3d_conformer(mol)
        if not success:
            msg = f"Failed to compute 3D conformer for {smiles}"
            raise ValueError(msg)
        return mol

    # Generate InChIKey for cache lookup
    try:
        mol_no_h = AllChem.RemoveHs(mol)
        inchi_key = Chem.MolToInChIKey(mol_no_h)
    except Exception as e:
        print(f"WARNING: Failed to generate InChIKey for {smiles}: {e}. Computing conformer without caching.")
        success = compute_3d_conformer(mol)
        if not success:
            msg = f"Failed to compute 3D conformer for {smiles}"
            raise ValueError(msg)
        return mol

    # Determine cache path based on affinity
    if affinity:
        conformer_cache_dir = cache_dir / "conformers" / "affinity"
    else:
        conformer_cache_dir = cache_dir / "conformers"

    conformer_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = conformer_cache_dir / f"{inchi_key}.pkl"

    # Try to load from cache
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cached_data = pickle.load(f)

            cached_mol = cached_data["mol"]
            cached_smiles = cached_data["smiles"]

            # Check for hash collision by comparing canonical SMILES
            mol_canonical = Chem.MolToSmiles(mol_no_h)
            cached_canonical = Chem.MolToSmiles(AllChem.RemoveHs(cached_mol))

            if mol_canonical != cached_canonical:
                print(
                    f"WARNING: InChIKey collision detected for {inchi_key}!\n"
                    f"  Query SMILES: {smiles} (canonical: {mol_canonical})\n"
                    f"  Cached SMILES: {cached_smiles} (canonical: {cached_canonical})\n"
                    f"  Overwriting cache with new molecule."
                )
            else:
                # Cache hit with matching molecule
                return cached_mol

        except Exception as e:
            print(f"WARNING: Failed to load cached conformer from {cache_path}: {e}. Regenerating.")

    # Cache miss or collision - compute conformer
    success = compute_3d_conformer(mol)
    if not success:
        msg = f"Failed to compute 3D conformer for {smiles}"
        raise ValueError(msg)

    # Save to cache
    try:
        cache_data = {
            "mol": mol,
            "smiles": smiles,
        }
        with cache_path.open("wb") as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        print(f"WARNING: Failed to cache conformer to {cache_path}: {e}")

    return mol
