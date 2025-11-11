"""Caching functionality for conformers, parsed polymers, templates, and MSAs."""

import hashlib
import pickle
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from rdkit import Chem
from rdkit.Chem import AllChem

if TYPE_CHECKING:
    from boltz.data.parse.schema import ParsedChain
    from boltz.data.parse.mmcif import ParsedStructure
    from boltz.data.types import MSA

LRU_CACHE_SIZE = 64


@lru_cache(maxsize=LRU_CACHE_SIZE)
def _load_conformer_from_cache(
    smiles: str,
    affinity: bool,
    cache_dir_str: Optional[str],
) -> Chem.Mol:
    """Load conformer from disk cache. LRU cached in-memory.

    Parameters
    ----------
    smiles : str
        The SMILES string of the molecule
    affinity : bool
        Whether this molecule is used for affinity prediction
    cache_dir_str : str, optional
        String path to cache directory. If None, no caching is performed.

    Returns
    -------
    Chem.Mol
        The molecule with 3D conformer

    Raises
    ------
    ValueError
        If conformer generation fails
    """
    # Late import to avoid circular dependency
    from boltz.data.parse.schema import compute_3d_conformer

    # Create mol with hydrogens
    mol = AllChem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)

    # If no cache directory, just compute conformer
    if cache_dir_str is None:
        success = compute_3d_conformer(mol)
        if not success:
            msg = f"Failed to compute 3D conformer for {smiles}"
            raise ValueError(msg)
        return mol

    cache_dir = Path(cache_dir_str)

    # Generate cache hash from SMILES + affinity flag
    try:
        cache_key_data = (smiles, affinity)
        cache_key_str = str(cache_key_data).encode("utf-8")
        cache_hash = hashlib.sha256(cache_key_str).hexdigest()[:16]
    except Exception as e:
        print(f"WARNING: Failed to generate cache key for {smiles}: {e}. Computing conformer without caching.")
        success = compute_3d_conformer(mol)
        if not success:
            msg = f"Failed to compute 3D conformer for {smiles}"
            raise ValueError(msg)
        return mol

    # Use single conformers directory (affinity flag already in hash)
    conformer_cache_dir = cache_dir / "conformers"
    conformer_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = conformer_cache_dir / f"{cache_hash}.pkl"

    # Try to load from cache
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cached_data = pickle.load(f)

            cached_mol = cached_data["mol"]
            cached_smiles = cached_data["smiles"]
            cached_affinity = cached_data["affinity"]

            # Check for hash collision by comparing SMILES and affinity
            if smiles != cached_smiles or affinity != cached_affinity:
                print(
                    f"WARNING: Hash collision detected for {cache_hash}!\n"
                    f"  Query: smiles={smiles}, affinity={affinity}\n"
                    f"  Cached: smiles={cached_smiles}, affinity={cached_affinity}\n"
                    f"  Overwriting cache with new molecule."
                )
            else:
                # Cache hit with matching molecule
                smiles_short = smiles[:50] + "..." if len(smiles) > 50 else smiles
                print(f"Using cached conformer: {smiles_short}")
                return cached_mol

        except Exception as e:
            print(f"WARNING: Failed to load cached conformer from {cache_path}: {e}. Regenerating.")

    # Cache miss or collision - compute conformer
    smiles_short = smiles[:50] + "..." if len(smiles) > 50 else smiles
    print(f"Computing conformer: {smiles_short}")
    success = compute_3d_conformer(mol)
    if not success:
        msg = f"Failed to compute 3D conformer for {smiles}"
        raise ValueError(msg)

    # Save to cache
    try:
        cache_data = {
            "mol": mol,
            "smiles": smiles,
            "affinity": affinity,
        }
        with cache_path.open("wb") as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        print(f"WARNING: Failed to cache conformer to {cache_path}: {e}")

    return mol


def get_mol_with_conformer(
    smiles: str,
    affinity: bool,
    cache_dir: Optional[Path] = None,
) -> Chem.Mol:
    """Get a molecule with 3D conformer, using cache if available.

    This function caches conformers based on a hash of (SMILES, affinity).

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
    cache_dir_str = str(cache_dir) if cache_dir is not None else None
    return _load_conformer_from_cache(smiles, affinity, cache_dir_str)


@lru_cache(maxsize=LRU_CACHE_SIZE)
def _load_polymer_from_cache(
    sequence_tuple: tuple[str, ...],
    chain_type: int,
    cyclic: bool,
    cache_dir_str: Optional[str],
) -> Optional["ParsedChain"]:
    """Load polymer from disk cache. LRU cached in-memory.

    Parameters
    ----------
    sequence_tuple : tuple[str, ...]
        The sequence of residue names as a tuple
    chain_type : int
        The chain type ID (protein/DNA/RNA)
    cyclic : bool
        Whether the chain is cyclic
    cache_dir_str : str, optional
        String path to cache directory. If None, returns None.

    Returns
    -------
    ParsedChain, optional
        The cached parsed chain object, or None if not cached

    """
    # If no cache directory, return None (no cache available)
    if cache_dir_str is None:
        return None

    cache_dir = Path(cache_dir_str)

    # Generate SHA256 hash for cache lookup
    try:
        cache_key_data = (sequence_tuple, chain_type, cyclic)
        cache_key_str = str(cache_key_data).encode("utf-8")
        # Use first 16 characters for shorter filenames
        cache_hash = hashlib.sha256(cache_key_str).hexdigest()[:16]
    except Exception as e:
        print(f"WARNING: Failed to generate cache key for polymer: {e}.")
        return None

    # Determine cache path
    polymer_cache_dir = cache_dir / "polymers"
    polymer_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = polymer_cache_dir / f"{cache_hash}.pkl"

    # Try to load from cache
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cached_data = pickle.load(f)

            cached_chain = cached_data["chain"]
            cached_sequence = cached_data["sequence"]
            cached_chain_type = cached_data["chain_type"]
            cached_cyclic = cached_data["cyclic"]

            # Check for hash collision by comparing inputs
            if (
                sequence_tuple == tuple(cached_sequence)
                and chain_type == cached_chain_type
                and cyclic == cached_cyclic
            ):
                # Cache hit with matching polymer
                seq_str = "".join(cached_sequence[:20])
                if len(cached_sequence) > 20:
                    seq_str += f"... ({len(cached_sequence)} residues)"
                print(f"Using cached polymer: {seq_str}")
                return cached_chain
            else:
                print(
                    f"WARNING: Cache collision detected for hash {cache_hash}!\n"
                    f"  Query: sequence_len={len(sequence_tuple)}, chain_type={chain_type}, cyclic={cyclic}\n"
                    f"  Cached: sequence_len={len(cached_sequence)}, chain_type={cached_chain_type}, cyclic={cached_cyclic}\n"
                    f"  Will overwrite cache with new polymer."
                )
                return None

        except Exception as e:
            print(f"WARNING: Failed to load cached polymer from {cache_path}: {e}.")
            return None

    # Cache miss
    return None


def get_polymer_with_cache(
    sequence: list[str],
    raw_sequence: str,
    entity: str,
    chain_type: int,
    components: dict[str, Chem.Mol],
    cyclic: bool,
    mol_dir: Path,
    cache_dir: Optional[Path] = None,
) -> Optional["ParsedChain"]:
    """Get a parsed polymer chain, using cache if available.

    This function caches parsed polymer chains based on a SHA256 hash of
    (sequence, chain_type, cyclic).

    Parameters
    ----------
    sequence : list[str]
        The sequence of residue names
    raw_sequence : str
        The raw sequence string
    entity : str
        The entity ID
    chain_type : int
        The chain type ID (protein/DNA/RNA)
    components : dict[str, Mol]
        The CCD components dictionary
    cyclic : bool
        Whether the chain is cyclic
    mol_dir : Path
        Path to the molecule directory
    cache_dir : Path, optional
        The cache directory. If None, no caching is performed.

    Returns
    -------
    ParsedChain, optional
        The parsed chain object

    """
    # Late import to avoid circular dependency
    from boltz.data.parse.schema import parse_polymer

    # Convert to hashable types for cache lookup
    sequence_tuple = tuple(sequence)
    cache_dir_str = str(cache_dir) if cache_dir is not None else None

    # Try to load from cache
    cached_chain = _load_polymer_from_cache(
        sequence_tuple, chain_type, cyclic, cache_dir_str
    )
    if cached_chain is not None:
        # Update the entity field to match the current context
        # The cached chain may have a different entity ID from a previous parse
        return replace(cached_chain, entity=entity)

    # Cache miss - parse the polymer
    seq_str = raw_sequence[:20]
    if len(raw_sequence) > 20:
        seq_str += f"... ({len(sequence)} residues)"
    print(f"Parsing polymer: {seq_str}")
    parsed_chain = parse_polymer(
        sequence=sequence,
        raw_sequence=raw_sequence,
        entity=entity,
        chain_type=chain_type,
        components=components,
        cyclic=cyclic,
        mol_dir=mol_dir,
    )

    # Save to disk cache
    if parsed_chain is not None and cache_dir is not None:
        try:
            # Recreate cache path (same logic as in _load_polymer_from_cache)
            cache_key_data = (sequence_tuple, chain_type, cyclic)
            cache_key_str = str(cache_key_data).encode("utf-8")
            # Use first 16 characters for shorter filenames
            cache_hash = hashlib.sha256(cache_key_str).hexdigest()[:16]

            polymer_cache_dir = cache_dir / "polymers"
            polymer_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = polymer_cache_dir / f"{cache_hash}.pkl"

            cache_data = {
                "chain": parsed_chain,
                "sequence": sequence,
                "chain_type": chain_type,
                "cyclic": cyclic,
            }
            with cache_path.open("wb") as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"WARNING: Failed to cache polymer: {e}")

    return parsed_chain


def _compute_file_checksum(file_path: Path) -> str:
    """Compute SHA256 checksum of a file.

    Parameters
    ----------
    file_path : Path
        The path to the file

    Returns
    -------
    str
        The SHA256 checksum (hex string)

    """
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


@lru_cache(maxsize=LRU_CACHE_SIZE)
def _load_template_from_cache(
    template_path_str: str,
    use_assembly: bool,
    compute_interfaces: bool,
    cache_dir_str: Optional[str],
) -> Optional["ParsedStructure"]:
    """Load template from disk cache. LRU cached in-memory.

    Note: Checksum validation only happens at disk read stage, not for LRU lookups.
    We assume files won't change during a single run.

    Parameters
    ----------
    template_path_str : str
        The absolute path to the template file
    use_assembly : bool
        Whether to use biological assembly
    compute_interfaces : bool
        Whether to compute interfaces
    cache_dir_str : str, optional
        String path to cache directory. If None, returns None.

    Returns
    -------
    ParsedStructure, optional
        The cached parsed template, or None if not cached

    """
    # If no cache directory, return None (no cache available)
    if cache_dir_str is None:
        return None

    cache_dir = Path(cache_dir_str)
    template_path_obj = Path(template_path_str)

    # Generate SHA256 hash for cache lookup (without checksum in key for LRU)
    try:
        cache_key_data = (template_path_str, use_assembly, compute_interfaces)
        cache_key_str = str(cache_key_data).encode("utf-8")
        # Use first 16 characters for shorter filenames
        cache_hash = hashlib.sha256(cache_key_str).hexdigest()[:16]
    except Exception as e:
        print(f"WARNING: Failed to generate cache key for template: {e}.")
        return None

    # Determine cache path
    template_cache_dir = cache_dir / "templates"
    template_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = template_cache_dir / f"{cache_hash}.pkl"

    # Try to load from cache
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cached_data = pickle.load(f)

            cached_template = cached_data["template"]
            cached_path = cached_data["path"]
            cached_checksum = cached_data["checksum"]
            cached_use_assembly = cached_data["use_assembly"]
            cached_compute_interfaces = cached_data["compute_interfaces"]

            # Verify checksum to detect file changes (only at disk read stage)
            current_checksum = _compute_file_checksum(template_path_obj)

            # Check for hash collision or file modification by comparing inputs
            if (
                template_path_str == cached_path
                and current_checksum == cached_checksum
                and use_assembly == cached_use_assembly
                and compute_interfaces == cached_compute_interfaces
            ):
                # Cache hit with matching template
                print(f"Using cached template: {template_path_obj.name}")
                return cached_template
            else:
                print(
                    f"WARNING: Cache collision or template file modification detected for hash {cache_hash}!\n"
                    f"  Query: path={template_path_str}, checksum={current_checksum}, use_assembly={use_assembly}, compute_interfaces={compute_interfaces}\n"
                    f"  Cached: path={cached_path}, checksum={cached_checksum}, use_assembly={cached_use_assembly}, compute_interfaces={cached_compute_interfaces}\n"
                    f"  Will overwrite cache with new template."
                )
                return None

        except Exception as e:
            print(f"WARNING: Failed to load cached template from {cache_path}: {e}.")
            return None

    # Cache miss
    return None


def get_template_with_cache(
    template_path: str,
    mols: dict[str, Chem.Mol],
    moldir: Optional[str],
    use_assembly: bool,
    compute_interfaces: bool,
    cache_dir: Optional[Path] = None,
) -> "ParsedStructure":
    """Get a parsed template structure, using cache if available.

    This function caches parsed templates. Checksums are computed only at disk
    read stage to detect file changes between runs.

    Parameters
    ----------
    template_path : str
        The path to the template file (PDB or CIF)
    mols : dict[str, Mol]
        The CCD components dictionary
    moldir : str, optional
        Path to the molecule directory
    use_assembly : bool
        Whether to use biological assembly
    compute_interfaces : bool
        Whether to compute interfaces
    cache_dir : Path, optional
        The cache directory. If None, no caching is performed.

    Returns
    -------
    ParsedStructure
        The parsed template structure

    """
    # Late imports to avoid circular dependency
    from boltz.data.parse.pdb import parse_pdb
    from boltz.data.parse.mmcif import parse_mmcif

    # Get absolute path
    template_path_obj = Path(template_path).resolve()
    template_path_str = str(template_path_obj)

    # Convert to hashable types for cache lookup
    cache_dir_str = str(cache_dir) if cache_dir is not None else None

    # Try to load from cache (checksum validation happens inside at disk read)
    cached_template = _load_template_from_cache(
        template_path_str, use_assembly, compute_interfaces, cache_dir_str
    )
    if cached_template is not None:
        return cached_template

    # Cache miss - parse the template
    print(f"Parsing template: {template_path_obj.name}")
    is_pdb = template_path_obj.suffix.lower() in {".pdb", ".ent"}

    if is_pdb:
        parsed_template = parse_pdb(
            path=template_path,
            mols=mols,
            moldir=moldir,
            use_assembly=use_assembly,
            compute_interfaces=compute_interfaces,
        )
    else:
        parsed_template = parse_mmcif(
            path=template_path,
            mols=mols,
            moldir=moldir,
            use_assembly=use_assembly,
            compute_interfaces=compute_interfaces,
        )

    # Save to disk cache (compute checksum for storage)
    if cache_dir is not None:
        try:
            # Compute checksum for cache storage
            checksum = _compute_file_checksum(template_path_obj)

            # Recreate cache path (same logic as in _load_template_from_cache)
            cache_key_data = (template_path_str, use_assembly, compute_interfaces)
            cache_key_str = str(cache_key_data).encode("utf-8")
            # Use first 16 characters for shorter filenames
            cache_hash = hashlib.sha256(cache_key_str).hexdigest()[:16]

            template_cache_dir = cache_dir / "templates"
            template_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = template_cache_dir / f"{cache_hash}.pkl"

            cache_data = {
                "template": parsed_template,
                "path": template_path_str,
                "checksum": checksum,
                "use_assembly": use_assembly,
                "compute_interfaces": compute_interfaces,
            }
            with cache_path.open("wb") as f:
                pickle.dump(cache_data, f)
            print(f"Cached template: {template_path_obj.name}")
        except Exception as e:
            print(f"WARNING: Failed to cache template: {e}")

    return parsed_template


@lru_cache(maxsize=LRU_CACHE_SIZE)
def _load_msa_from_cache(
    msa_path_str: str,
    max_seqs: Optional[int],
    cache_dir_str: Optional[str],
) -> Optional["MSA"]:
    """Load MSA from disk cache. LRU cached in-memory.

    Note: Checksum validation only happens at disk read stage, not for LRU lookups.
    We assume files won't change during a single run.

    Parameters
    ----------
    msa_path_str : str
        The absolute path to the MSA file
    max_seqs : int, optional
        Maximum number of sequences to include
    cache_dir_str : str, optional
        String path to cache directory. If None, returns None.

    Returns
    -------
    MSA, optional
        The cached parsed MSA, or None if not cached

    """
    # If no cache directory, return None (no cache available)
    if cache_dir_str is None:
        return None

    cache_dir = Path(cache_dir_str)
    msa_path_obj = Path(msa_path_str)

    # Generate SHA256 hash for cache lookup (without checksum in key for LRU)
    try:
        cache_key_data = (msa_path_str, max_seqs)
        cache_key_str = str(cache_key_data).encode("utf-8")
        # Use first 16 characters for shorter filenames
        cache_hash = hashlib.sha256(cache_key_str).hexdigest()[:16]
    except Exception as e:
        print(f"WARNING: Failed to generate cache key for MSA: {e}.")
        return None

    # Determine cache path
    msa_cache_dir = cache_dir / "msa"
    msa_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = msa_cache_dir / f"{cache_hash}.npz"

    # Try to load from cache
    if cache_path.exists():
        try:
            # Load from NPZ (MSA has a load method from NumpySerializable)
            from boltz.data.types import MSA

            cached_msa = MSA.load(cache_path)

            # Verify checksum to detect file changes (only at disk read stage)
            # Read cached metadata from a separate pickle file
            metadata_path = msa_cache_dir / f"{cache_hash}.meta"
            if metadata_path.exists():
                with metadata_path.open("rb") as f:
                    cached_meta = pickle.load(f)

                cached_path = cached_meta["path"]
                cached_checksum = cached_meta["checksum"]
                cached_max_seqs = cached_meta["max_seqs"]

                # Verify checksum
                current_checksum = _compute_file_checksum(msa_path_obj)

                # Check for hash collision or file modification by comparing inputs
                if (
                    msa_path_str == cached_path
                    and current_checksum == cached_checksum
                    and max_seqs == cached_max_seqs
                ):
                    # Cache hit with matching MSA
                    print(f"Using cached MSA: {msa_path_obj.name}")
                    return cached_msa
                else:
                    print(
                        f"WARNING: Cache collision or MSA file modification detected for hash {cache_hash}!\n"
                        f"  Query: path={msa_path_str}, checksum={current_checksum}, max_seqs={max_seqs}\n"
                        f"  Cached: path={cached_path}, checksum={cached_checksum}, max_seqs={cached_max_seqs}\n"
                        f"  Will overwrite cache with new MSA."
                    )
                    return None
            else:
                # No metadata file - cache invalid
                print(f"WARNING: MSA cache metadata missing for {cache_hash}. Re-parsing.")
                return None

        except Exception as e:
            print(f"WARNING: Failed to load cached MSA from {cache_path}: {e}.")
            return None

    # Cache miss
    return None


def get_msa_with_cache(
    msa_path: Path,
    max_seqs: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> "MSA":
    """Get a parsed MSA, using cache if available.

    This function caches parsed MSAs. Checksums are computed only at disk
    read stage to detect file changes between runs.

    Parameters
    ----------
    msa_path : Path
        The path to the MSA file (.a3m, .a3m.gz, or .csv)
    max_seqs : int, optional
        Maximum number of sequences to include
    cache_dir : Path, optional
        The cache directory. If None, no caching is performed.

    Returns
    -------
    MSA
        The parsed MSA object

    """
    # Late imports to avoid circular dependency
    from boltz.data.parse.a3m import parse_a3m
    from boltz.data.parse.csv import parse_csv

    # Get absolute path
    msa_path_obj = Path(msa_path).resolve()
    msa_path_str = str(msa_path_obj)

    # Convert to hashable types for cache lookup
    cache_dir_str = str(cache_dir) if cache_dir is not None else None

    # Try to load from cache (checksum validation happens inside at disk read)
    cached_msa = _load_msa_from_cache(msa_path_str, max_seqs, cache_dir_str)
    if cached_msa is not None:
        return cached_msa

    # Cache miss - parse the MSA
    print(f"Parsing MSA: {msa_path_obj.name}")

    # Parse based on file type
    if msa_path_obj.suffix == ".csv":
        parsed_msa = parse_csv(msa_path_obj, max_seqs=max_seqs)
    elif msa_path_obj.suffix in {".a3m", ".gz"}:
        # taxonomy is always None in main.py, so we hardcode it
        parsed_msa = parse_a3m(msa_path_obj, taxonomy=None, max_seqs=max_seqs)
    else:
        msg = f"MSA file {msa_path_obj} not supported, only .a3m, .a3m.gz, or .csv."
        raise ValueError(msg)

    # Save to disk cache
    if cache_dir is not None:
        try:
            # Compute checksum for cache storage
            checksum = _compute_file_checksum(msa_path_obj)

            # Recreate cache path (same logic as in _load_msa_from_cache)
            cache_key_data = (msa_path_str, max_seqs)
            cache_key_str = str(cache_key_data).encode("utf-8")
            # Use first 16 characters for shorter filenames
            cache_hash = hashlib.sha256(cache_key_str).hexdigest()[:16]

            msa_cache_dir = cache_dir / "msa"
            msa_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = msa_cache_dir / f"{cache_hash}.npz"
            metadata_path = msa_cache_dir / f"{cache_hash}.meta"

            # Save MSA to NPZ
            parsed_msa.dump(cache_path)

            # Save metadata to pickle
            cache_meta = {
                "path": msa_path_str,
                "checksum": checksum,
                "max_seqs": max_seqs,
            }
            with metadata_path.open("wb") as f:
                pickle.dump(cache_meta, f)

            print(f"Cached MSA: {msa_path_obj.name}")
        except Exception as e:
            print(f"WARNING: Failed to cache MSA: {e}")

    return parsed_msa
