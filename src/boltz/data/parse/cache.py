"""Caching functionality for conformers, parsed polymers, templates, and MSAs."""

import hashlib
import pickle
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from filelock import FileLock
from rdkit import Chem
from rdkit.Chem import AllChem

if TYPE_CHECKING:
    from boltz.data.parse.schema import ParsedChain
    from boltz.data.parse.mmcif import ParsedStructure
    from boltz.data.types import MSA

LRU_CACHE_SIZE = 64
CACHE_LOCK_TIMEOUT = 60  # seconds


def _get_cache_lock(cache_path: Path, timeout: int = CACHE_LOCK_TIMEOUT) -> FileLock:
    """Get a file lock for a specific cache file.

    Parameters
    ----------
    cache_path : Path
        The path to the cache file
    timeout : int
        Lock timeout in seconds (default: 60)

    Returns
    -------
    FileLock
        A file lock for process-safe synchronization

    """
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    return FileLock(lock_path, timeout=timeout)


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

    # Fast path: try to load from cache without lock
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
                # Continue to locked section to overwrite
            else:
                # Cache hit with matching molecule
                smiles_short = smiles[:50] + "..." if len(smiles) > 50 else smiles
                print(f"Using cached conformer: {smiles_short}")
                return cached_mol

        except Exception as e:
            print(f"WARNING: Failed to load cached conformer from {cache_path}: {e}. Regenerating.")
            # Continue to locked section to regenerate

    # Cache miss or collision - use per-key file lock to prevent race conditions
    cache_lock = _get_cache_lock(cache_path)

    with cache_lock:
        # Double-check: cache might have been populated while waiting for lock
        if cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    cached_data = pickle.load(f)

                cached_mol = cached_data["mol"]
                cached_smiles = cached_data["smiles"]
                cached_affinity = cached_data["affinity"]

                # Check for hash collision
                if smiles == cached_smiles and affinity == cached_affinity:
                    # Cache was populated by another thread - use it
                    smiles_short = smiles[:50] + "..." if len(smiles) > 50 else smiles
                    print(f"Using cached conformer: {smiles_short}")
                    return cached_mol
                # If collision, continue to recompute and overwrite

            except Exception:
                # Failed to load - continue to recompute
                pass

        # Cache still doesn't exist or is invalid - compute conformer
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


def _get_polymer_internal(
    sequence_tuple: tuple[str, ...],
    raw_sequence: str,
    entity: str,
    chain_type: int,
    components: dict[str, Chem.Mol],
    cyclic: bool,
    mol_dir: Path,
    cache_dir: Optional[Path],
) -> Optional["ParsedChain"]:
    """Internal function to get parsed polymer with disk cache and file locks.

    This function handles disk caching with file locks for process-safe
    synchronization. It is called by get_polymer_with_cache.

    Parameters
    ----------
    sequence_tuple : tuple[str, ...]
        The sequence of residue names as a tuple
    raw_sequence : str
        The raw sequence string for display
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

    # If no cache directory, just parse directly
    if cache_dir is None:
        seq_str = raw_sequence[:20]
        if len(raw_sequence) > 20:
            seq_str += f"... ({len(sequence_tuple)} residues)"
        print(f"Parsing polymer: {seq_str}")
        return parse_polymer(
            sequence=list(sequence_tuple),
            raw_sequence=raw_sequence,
            entity=entity,
            chain_type=chain_type,
            components=components,
            cyclic=cyclic,
            mol_dir=mol_dir,
        )

    # Generate cache hash
    try:
        cache_key_data = (sequence_tuple, chain_type, cyclic)
        cache_key_str = str(cache_key_data).encode("utf-8")
        cache_hash = hashlib.sha256(cache_key_str).hexdigest()[:16]
    except Exception as e:
        print(f"WARNING: Failed to generate cache key for polymer: {e}. Parsing without caching.")
        seq_str = raw_sequence[:20]
        if len(raw_sequence) > 20:
            seq_str += f"... ({len(sequence_tuple)} residues)"
        print(f"Parsing polymer: {seq_str}")
        return parse_polymer(
            sequence=list(sequence_tuple),
            raw_sequence=raw_sequence,
            entity=entity,
            chain_type=chain_type,
            components=components,
            cyclic=cyclic,
            mol_dir=mol_dir,
        )

    # Determine cache path
    polymer_cache_dir = cache_dir / "polymers"
    polymer_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = polymer_cache_dir / f"{cache_hash}.pkl"

    # Fast path: try to load from cache without lock
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cached_data = pickle.load(f)

            cached_chain = cached_data["chain"]
            cached_sequence = cached_data["sequence"]
            cached_chain_type = cached_data["chain_type"]
            cached_cyclic = cached_data["cyclic"]

            # Check for hash collision
            if (
                sequence_tuple == tuple(cached_sequence)
                and chain_type == cached_chain_type
                and cyclic == cached_cyclic
            ):
                # Cache hit - update entity and return
                seq_str = "".join(cached_sequence[:20])
                if len(cached_sequence) > 20:
                    seq_str += f"... ({len(cached_sequence)} residues)"
                print(f"Using cached polymer: {seq_str}")
                return replace(cached_chain, entity=entity)
            else:
                print(
                    f"WARNING: Cache collision detected for hash {cache_hash}!\n"
                    f"  Query: sequence_len={len(sequence_tuple)}, chain_type={chain_type}, cyclic={cyclic}\n"
                    f"  Cached: sequence_len={len(cached_sequence)}, chain_type={cached_chain_type}, cyclic={cached_cyclic}\n"
                    f"  Overwriting cache with new polymer."
                )
                # Continue to locked section to overwrite

        except Exception as e:
            print(f"WARNING: Failed to load cached polymer from {cache_path}: {e}.")
            # Continue to locked section to regenerate

    # Cache miss or collision - use file lock for process-safe synchronization
    cache_lock = _get_cache_lock(cache_path)

    with cache_lock:
        # Double-check: cache might have been populated while waiting for lock
        if cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    cached_data = pickle.load(f)

                cached_chain = cached_data["chain"]
                cached_sequence = cached_data["sequence"]
                cached_chain_type = cached_data["chain_type"]
                cached_cyclic = cached_data["cyclic"]

                # Check for hash collision
                if (
                    sequence_tuple == tuple(cached_sequence)
                    and chain_type == cached_chain_type
                    and cyclic == cached_cyclic
                ):
                    # Cache was populated by another process - use it
                    seq_str = "".join(cached_sequence[:20])
                    if len(cached_sequence) > 20:
                        seq_str += f"... ({len(cached_sequence)} residues)"
                    print(f"Using cached polymer: {seq_str}")
                    return replace(cached_chain, entity=entity)
                # If collision, continue to recompute and overwrite

            except Exception:
                # Failed to load - continue to recompute
                pass

        # Cache still doesn't exist or is invalid - parse the polymer
        seq_str = raw_sequence[:20]
        if len(raw_sequence) > 20:
            seq_str += f"... ({len(sequence_tuple)} residues)"
        print(f"Parsing polymer: {seq_str}")
        parsed_chain = parse_polymer(
            sequence=list(sequence_tuple),
            raw_sequence=raw_sequence,
            entity=entity,
            chain_type=chain_type,
            components=components,
            cyclic=cyclic,
            mol_dir=mol_dir,
        )

        # Save to disk cache
        if parsed_chain is not None:
            try:
                cache_data = {
                    "chain": parsed_chain,
                    "sequence": list(sequence_tuple),
                    "chain_type": chain_type,
                    "cyclic": cyclic,
                }
                with cache_path.open("wb") as f:
                    pickle.dump(cache_data, f)
            except Exception as e:
                print(f"WARNING: Failed to cache polymer: {e}")

    return parsed_chain


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
    # Convert to hashable types
    sequence_tuple = tuple(sequence)

    return _get_polymer_internal(
        sequence_tuple=sequence_tuple,
        raw_sequence=raw_sequence,
        entity=entity,
        chain_type=chain_type,
        components=components,
        cyclic=cyclic,
        mol_dir=mol_dir,
        cache_dir=cache_dir,
    )


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


def get_template_with_cache(
    template_path: str,
    mols: dict[str, Chem.Mol],
    moldir: Optional[str],
    use_assembly: bool,
    compute_interfaces: bool,
    cache_dir: Optional[Path] = None,
) -> "ParsedStructure":
    """Get a parsed template structure, using cache if available.

    This function caches parsed templates with file locks for process-safe
    synchronization. Checksums are validated to detect file changes between runs.

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

    # If no cache directory, just parse directly
    if cache_dir is None:
        print(f"Parsing template: {template_path_obj.name}")
        is_pdb = template_path_obj.suffix.lower() in {".pdb", ".ent"}

        if is_pdb:
            return parse_pdb(
                path=template_path,
                mols=mols,
                moldir=moldir,
                use_assembly=use_assembly,
                compute_interfaces=compute_interfaces,
            )
        else:
            return parse_mmcif(
                path=template_path,
                mols=mols,
                moldir=moldir,
                use_assembly=use_assembly,
                compute_interfaces=compute_interfaces,
            )

    # Generate cache hash
    try:
        cache_key_data = (template_path_str, use_assembly, compute_interfaces)
        cache_key_str = str(cache_key_data).encode("utf-8")
        cache_hash = hashlib.sha256(cache_key_str).hexdigest()[:16]
    except Exception as e:
        print(f"WARNING: Failed to generate cache key for template: {e}. Parsing without caching.")
        print(f"Parsing template: {template_path_obj.name}")
        is_pdb = template_path_obj.suffix.lower() in {".pdb", ".ent"}

        if is_pdb:
            return parse_pdb(
                path=template_path,
                mols=mols,
                moldir=moldir,
                use_assembly=use_assembly,
                compute_interfaces=compute_interfaces,
            )
        else:
            return parse_mmcif(
                path=template_path,
                mols=mols,
                moldir=moldir,
                use_assembly=use_assembly,
                compute_interfaces=compute_interfaces,
            )

    # Determine cache path
    template_cache_dir = cache_dir / "templates"
    template_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = template_cache_dir / f"{cache_hash}.pkl"

    # Fast path: try to load from cache without lock
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cached_data = pickle.load(f)

            cached_template = cached_data["template"]
            cached_path = cached_data["path"]
            cached_checksum = cached_data["checksum"]
            cached_use_assembly = cached_data["use_assembly"]
            cached_compute_interfaces = cached_data["compute_interfaces"]

            # Verify checksum to detect file changes
            current_checksum = _compute_file_checksum(template_path_obj)

            # Check for hash collision or file modification
            if (
                template_path_str == cached_path
                and current_checksum == cached_checksum
                and use_assembly == cached_use_assembly
                and compute_interfaces == cached_compute_interfaces
            ):
                # Cache hit
                print(f"Using cached template: {template_path_obj.name}")
                return cached_template
            else:
                print(
                    f"WARNING: Cache collision or template file modification detected for hash {cache_hash}!\n"
                    f"  Query: path={template_path_str}, checksum={current_checksum}, use_assembly={use_assembly}, compute_interfaces={compute_interfaces}\n"
                    f"  Cached: path={cached_path}, checksum={cached_checksum}, use_assembly={cached_use_assembly}, compute_interfaces={cached_compute_interfaces}\n"
                    f"  Overwriting cache with new template."
                )
                # Continue to locked section to overwrite

        except Exception as e:
            print(f"WARNING: Failed to load cached template from {cache_path}: {e}.")
            # Continue to locked section to regenerate

    # Cache miss or collision - use file lock for process-safe synchronization
    cache_lock = _get_cache_lock(cache_path)

    with cache_lock:
        # Double-check: cache might have been populated while waiting for lock
        if cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    cached_data = pickle.load(f)

                cached_template = cached_data["template"]
                cached_path = cached_data["path"]
                cached_checksum = cached_data["checksum"]
                cached_use_assembly = cached_data["use_assembly"]
                cached_compute_interfaces = cached_data["compute_interfaces"]

                # Verify checksum
                current_checksum = _compute_file_checksum(template_path_obj)

                # Check for hash collision or file modification
                if (
                    template_path_str == cached_path
                    and current_checksum == cached_checksum
                    and use_assembly == cached_use_assembly
                    and compute_interfaces == cached_compute_interfaces
                ):
                    # Cache was populated by another process - use it
                    print(f"Using cached template: {template_path_obj.name}")
                    return cached_template
                # If collision or modified, continue to recompute and overwrite

            except Exception:
                # Failed to load - continue to recompute
                pass

        # Cache still doesn't exist or is invalid - parse the template
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

        # Save to disk cache with checksum
        try:
            checksum = _compute_file_checksum(template_path_obj)

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


def get_msa_with_cache(
    msa_path: Path,
    max_seqs: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> "MSA":
    """Get a parsed MSA, using cache if available.

    This function caches parsed MSAs with file locks for process-safe
    synchronization. Checksums are validated to detect file changes between runs.

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
    from boltz.data.types import MSA

    # Get absolute path
    msa_path_obj = Path(msa_path).resolve()
    msa_path_str = str(msa_path_obj)

    # If no cache directory, just parse directly
    if cache_dir is None:
        print(f"Parsing MSA: {msa_path_obj.name}")

        if msa_path_obj.suffix == ".csv":
            return parse_csv(msa_path_obj, max_seqs=max_seqs)
        elif msa_path_obj.suffix in {".a3m", ".gz"}:
            return parse_a3m(msa_path_obj, taxonomy=None, max_seqs=max_seqs)
        else:
            msg = f"MSA file {msa_path_obj} not supported, only .a3m, .a3m.gz, or .csv."
            raise ValueError(msg)

    # Generate cache hash
    try:
        cache_key_data = (msa_path_str, max_seqs)
        cache_key_str = str(cache_key_data).encode("utf-8")
        cache_hash = hashlib.sha256(cache_key_str).hexdigest()[:16]
    except Exception as e:
        print(f"WARNING: Failed to generate cache key for MSA: {e}. Parsing without caching.")
        print(f"Parsing MSA: {msa_path_obj.name}")

        if msa_path_obj.suffix == ".csv":
            return parse_csv(msa_path_obj, max_seqs=max_seqs)
        elif msa_path_obj.suffix in {".a3m", ".gz"}:
            return parse_a3m(msa_path_obj, taxonomy=None, max_seqs=max_seqs)
        else:
            msg = f"MSA file {msa_path_obj} not supported, only .a3m, .a3m.gz, or .csv."
            raise ValueError(msg)

    # Determine cache paths
    msa_cache_dir = cache_dir / "msa"
    msa_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = msa_cache_dir / f"{cache_hash}.npz"
    metadata_path = msa_cache_dir / f"{cache_hash}.meta"

    # Fast path: try to load from cache without lock
    if cache_path.exists() and metadata_path.exists():
        try:
            # Load MSA from NPZ
            cached_msa = MSA.load(cache_path)

            # Load and verify metadata
            with metadata_path.open("rb") as f:
                cached_meta = pickle.load(f)

            cached_path = cached_meta["path"]
            cached_checksum = cached_meta["checksum"]
            cached_max_seqs = cached_meta["max_seqs"]

            # Verify checksum to detect file changes
            current_checksum = _compute_file_checksum(msa_path_obj)

            # Check for hash collision or file modification
            if (
                msa_path_str == cached_path
                and current_checksum == cached_checksum
                and max_seqs == cached_max_seqs
            ):
                # Cache hit
                print(f"Using cached MSA: {msa_path_obj.name}")
                return cached_msa
            else:
                print(
                    f"WARNING: Cache collision or MSA file modification detected for hash {cache_hash}!\n"
                    f"  Query: path={msa_path_str}, checksum={current_checksum}, max_seqs={max_seqs}\n"
                    f"  Cached: path={cached_path}, checksum={cached_checksum}, max_seqs={cached_max_seqs}\n"
                    f"  Overwriting cache with new MSA."
                )
                # Continue to locked section to overwrite

        except Exception as e:
            print(f"WARNING: Failed to load cached MSA from {cache_path}: {e}.")
            # Continue to locked section to regenerate

    # Cache miss or collision - use file lock for process-safe synchronization
    cache_lock = _get_cache_lock(cache_path)

    with cache_lock:
        # Double-check: cache might have been populated while waiting for lock
        if cache_path.exists() and metadata_path.exists():
            try:
                # Load MSA from NPZ
                cached_msa = MSA.load(cache_path)

                # Load and verify metadata
                with metadata_path.open("rb") as f:
                    cached_meta = pickle.load(f)

                cached_path = cached_meta["path"]
                cached_checksum = cached_meta["checksum"]
                cached_max_seqs = cached_meta["max_seqs"]

                # Verify checksum
                current_checksum = _compute_file_checksum(msa_path_obj)

                # Check for hash collision or file modification
                if (
                    msa_path_str == cached_path
                    and current_checksum == cached_checksum
                    and max_seqs == cached_max_seqs
                ):
                    # Cache was populated by another process - use it
                    print(f"Using cached MSA: {msa_path_obj.name}")
                    return cached_msa
                # If collision or modified, continue to recompute and overwrite

            except Exception:
                # Failed to load - continue to recompute
                pass

        # Cache still doesn't exist or is invalid - parse the MSA
        print(f"Parsing MSA: {msa_path_obj.name}")

        # Parse based on file type
        if msa_path_obj.suffix == ".csv":
            parsed_msa = parse_csv(msa_path_obj, max_seqs=max_seqs)
        elif msa_path_obj.suffix in {".a3m", ".gz"}:
            parsed_msa = parse_a3m(msa_path_obj, taxonomy=None, max_seqs=max_seqs)
        else:
            msg = f"MSA file {msa_path_obj} not supported, only .a3m, .a3m.gz, or .csv."
            raise ValueError(msg)

        # Save to disk cache with checksum
        try:
            checksum = _compute_file_checksum(msa_path_obj)

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
