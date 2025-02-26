# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
from argparse import ArgumentParser, Namespace

import psutil

# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import math
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from glob import glob
from typing import Iterable, Optional, cast

import numpy as np
from composer.utils import (
    ObjectStore,
    maybe_create_object_store_from_uri,
    parse_uri,
)
from numpy.typing import NDArray
from streaming import MDSWriter
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from torch.utils.data import DataLoader, IterableDataset

log = logging.getLogger(__name__)

DONE_FILENAME = '.text_to_mds_conversion_done'


class ConcatTokensFromFileDataset(IterableDataset):
    """An IterableDataset that returns token samples for MDSWriter from files.

    Returns dicts of {'tokens': ndarray:int32}

    Each line in the file is considered a sequence.
    """

    def __init__(
        self,
        file: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        bos_text: str,
        eos_text: str,
        no_wrap: bool,
    ):
        self.file = file
        # super().__init__(tokenizer, max_length, bos_text, eos_text, no_wrap)
        log.info(f'Initialized ConcatTokensFromFileDataset.')

    def __iter__(self) -> Iterable[dict[str, NDArray]]:
        buffer = []
        log.info(f'Processing file: {file}')
        with open(file, 'r') as f:
            for line in file:
                line = line.strip()  # Use .strip() to remove trailing newline characters
                buffer += self.bos_tokens
                encoded = self.tokenizer(
                            line,
                            truncation=False,
                            padding=False,
                        )
                iids = encoded['input_ids']

                # Add the tokens to the buffer
                buffer += iids
                # Add the EOS token to the buffer to separate lines.
                buffer += self.eos_tokens
                while len(buffer) >= self.max_length:
                    concat_sample = buffer[:self.max_length]
                    buffer = buffer[self.max_length:
                                    ] if self.should_wrap else []
                    yield {
                        'tokens':
                            np.asarray(concat_sample, dtype=np.int32),
                    }

        # Yield any remaining samples of size max_length.
        while len(buffer) >= self.max_length:
            concat_sample = buffer[:self.max_length]
            buffer = buffer[self.max_length:] if self.should_wrap else []
            yield {'tokens': np.asarray(concat_sample, dtype=np.int32)}

class NoConcatTokensFromFileDataset(IterableDataset):
    """An IterableDataset that returns token samples for MDSWriter from files.

    Returns dicts of {'tokens': ndarray:int32}

    Each line in the file is considered a sequence. No concatenation is done.
    """

    def __init__(
        self,
        file: str,
        tokenizer: PreTrainedTokenizerBase,
        bos_text: str,
        eos_text: str,
    ):
        self.file = file
        log.info(f'Initialized NoConcatTokensFromFileDataset.')

    def __iter__(self) -> Iterable[dict[str, NDArray]]:
        buffer = []
        log.info(f'Processing file: {self.file}')
        with open(self.file, 'r') as f:
            for line in f:
                line = line.strip()
                if line: 
                    yield {"text": line}


def download_and_convert_starargs(args: tuple):
    """Helper function to call download_and_convert with star args.

    This helps us use download_and_convert with multiprocessing.
    """
    return convert(*args)


def convert(
    output_folder: str,
    input_file: str,
    tokenizer_name: str,
    eos_text: str,
    bos_text: str,
    compression: str,
    trust_remote_code: bool,
):
    """Downloads and converts text files to MDS format.

    Args:
        file_names (List[str]): Files to process
        output_folder (str): Folder to write MDS shards to
        input_file (str): File to process
        tokenizer_name (str): Name of tokenizer to use
        eos_text (str): Text to append to each example to separate concatenated samples
        bos_text (str): Text to prepend to each example to separate concatenated samples
        compression (str): The compression algorithm to use for MDS writing
        trust_remote_code (bool): If true, allows custom code to be executed to load the tokenizer
    """
    object_store = maybe_create_object_store_from_uri(input_file)

    log.info(f'Initializing tokenizer: {tokenizer_name}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.model_max_length = 5000000000  # Hack to prevent warnings from HuggingFace

    # Use the ConcatTokensDataset from LLM-foundry to concatenate sequences of tokens up
    # to the maximum sequence length
    dataset = NoConcatTokensFromFileDataset(
        file=input_file,
        tokenizer=tokenizer,
        eos_text=eos_text,
        bos_text=bos_text,
    )

    columns = {'text': 'str'}

    log.info('Converting to MDS format...')
    with MDSWriter(
        out=output_folder,
        columns=columns,
        compression=compression,
    ) as out:
        for sample in tqdm(dataset):
            out.write(sample)

    log.info(f'Completed conversion to MDS format')


def is_remote_path(path: str) -> bool:
    """Checks whether a path is a remote path.

    Args:
        path (str): path to check
    """
    backend, _, _ = parse_uri(path)
    return backend != ''


def is_already_processed(
    output_root: str,
    args_str: str,
    object_names: list[str],
) -> bool:
    """Determines whether a group of text files has already been processed.

    Checks the done fie at output root to determine this.

    Args:
        output_root (str): Output folder where a done file may exist
        args_str (str): String representation of the arguments
        object_names (List[str]): Names of objects to convert to MDS format
    """
    log.info(
        f'Checking if {len(object_names)} objects have already been processed in {output_root}',
    )

    # Retrieve the done file contents
    output_object_store = maybe_create_object_store_from_uri(output_root)
    if output_object_store is not None:
        # Download and read the done file from the remote object store
        _, _, output_folder_prefix = parse_uri(output_root)
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                done_file = os.path.join(tmp_dir, DONE_FILENAME)
                download_file(
                    object_store=output_object_store,
                    object_name=os.path.join(
                        output_folder_prefix,
                        DONE_FILENAME,
                    ),
                    output_filename=done_file,
                )
                with open(done_file) as df:
                    done_file_contents = df.read().splitlines()
                log.info(f'Retrieved done file contents from remote storage')
        except FileNotFoundError:
            log.info('Done file not found in remote storage')
            return False
    else:
        # Read the local done file
        done_file = os.path.join(output_root, DONE_FILENAME)
        if not os.path.isfile(done_file):
            log.info('Done file not found in local storage')
            return False
        with open(done_file) as df:
            done_file_contents = df.read().splitlines()
        log.info(f'Retrieved done file contents from local storage')

    # Compare the arguments
    prev_args_str = done_file_contents[0]
    if prev_args_str != args_str:
        log.info('Arguments have changed, reprocessing required')
        return False

    # Compare file names
    prev_names = done_file_contents[1:]
    if len(prev_names) != len(object_names):
        log.info('Number of files has changed, reprocessing required')
        return False
    for idx, prev_name in enumerate(prev_names):
        if object_names[idx] != prev_name:
            log.info('File names have changed, reprocessing required')
            return False

    log.info('All files have already been processed')
    return True


def write_done_file(folder: str, args_str: str, object_names: list[str]):
    """Write a file to signify completion.

    This the done file includes the arguments to processing and
    a list of objects that were processed.

    Args:
        folder (str): Folder to write the done file to
        args_str (str): String representation of arguments
        object_names (List[str]): List of objects to convert to MDS format
    """
    with open(os.path.join(folder, DONE_FILENAME), 'w') as done_file:
        log.info(f'Writing done file.')
        done_file.write('\n'.join([args_str] + object_names) + '\n')
    log.info(f'Done file written successfully')


def convert_text_to_mds(
    tokenizer_name: str,
    output_folder: str,
    input_file: str,
    eos_text: str,
    bos_text: str,
    compression: str,
    processes: int,
    args_str: str,
    reprocess: bool,
    trust_remote_code: bool,
):
    """Convert a folder of text files to MDS format.

    Args:
        tokenizer_name (str): Name of tokenizer to use
        output_folder (str): Folder to write MDS shards to
        input_file (str): Wiki file to process
        concat_tokens (int): Concatenate up to this many tokens
        eos_text (str): Text to append to each example to separate concatenated samples
        bos_text (str): Text to prepend to each example to separate concatenated samples
        no_wrap: (bool): Whether to let text examples wrap across multiple training examples
        compression (str): The compression algorithm to use for MDS writing
        processes (int): The number of processes to use.
        args_str (str): String representation of the arguments
        reprocess (bool): Whether to always reprocess the given folder of text files
        trust_remote_code (bool): If true, allows custom code to be executed to load the tokenizer
    """
    # Load the tokenizer once on the main process so that the files are cached to avoid race conditions
    # in the Hugging Face load code
    AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
    )

    if os.path.isdir(output_folder) and len(os.listdir(output_folder)) > 0:
        log.error(f'Output folder is not empty: {output_folder}')
        raise Exception(f'Output folder is not empty: {output_folder}')

    log.info('Using single process for download and conversion')
    convert(
        output_folder,
        input_file,
        tokenizer_name,
        eos_text,
        bos_text,
        compression,
        trust_remote_code,
    )

    index_path = os.path.join(output_folder, 'index.json')
    with open(index_path, 'r') as index_file:
        if not json.load(index_file)['shards']:
            log.error('No shards were created when converting text to MDS.')

    # Write a done file with the args and object names
    write_done_file(output_folder, args_str, [input_file])


def _configure_logging(logging_level: str):
    """Configure logging.

    Args:
        logging_level (str): Logging level.
    """
    logging.basicConfig(
        format=
        f'%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s',
        force=True,
    )
    logging_level = logging_level.upper()
    logging.getLogger('llmfoundry').setLevel(logging_level)
    logging.getLogger(__name__).setLevel(logging_level)
    log.info(f'Logging level set to {logging_level}')


def convert_text_to_mds_from_args(
    output_folder: str,
    input_file: str,
    compression: str,
    tokenizer_name: str,
    bos_text: Optional[str],
    eos_text: Optional[str],
    use_tokenizer_eos: bool,
    processes: int,
    reprocess: bool,
    trust_remote_code: bool,
    logging_level: str,
) -> None:
    """A wrapper for `convert_text_to_mds` to parse arguments.

    Args:
        output_folder (str): Folder to write MDS shards to
        input_file (str): A wiki text file to process (one sequence per line)
        compression (str): The compression algorithm to use for MDS writing
        tokenizer_name (str): The name of the tokenizer to use
        bos_text (Optional[str]): The text to prepend to each example to separate concatenated examples
        eos_text (Optional[str]): The text to append to each example to separate concatenated examples
        use_tokenizer_eos (bool): Use the EOS text from the tokenizer
        processes (int): The number of processes to use to download and convert the dataset
        reprocess (bool): If true, reprocess the input_folder to MDS format. Otherwise, only reprocess upon changes to the input folder or dataset creation parameters.
        trust_remote_code (bool): If true, allows custom code to be executed to load the tokenizer
        logging_level (str): Logging level for the script. Default is INFO.

    Raises:
        ValueError: If `use_tokenizer_eos` is True and `eos_text` is not None
    """
    os.environ['WORLD_SIZE'] = '1'
    if use_tokenizer_eos:
        # Ensure that eos text is not specified twice.
        if eos_text is not None:
            ValueError(
                'Cannot set --eos_text with --use_tokenizer_eos. Please specify one.',
            )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
        )
        eos_text = tokenizer.eos_token

    # now that we have validated them, change BOS/EOS to strings
    if bos_text is None:
        bos_text = ''
    if eos_text is None:
        eos_text = ''
    _configure_logging(logging_level)

    # Define args for _args_str
    args = {
        'tokenizer': tokenizer_name,
        'output_folder': output_folder,
        'input_file': input_file,
        'compression': compression,
        'eos_text': eos_text,
        'bos_text': bos_text,
        'processes': processes,
        'reprocess': reprocess,
        'trust_remote_code': trust_remote_code,
    }
    convert_text_to_mds(
        tokenizer_name=tokenizer_name,
        output_folder=output_folder,
        input_file=input_file,
        eos_text=eos_text,
        bos_text=bos_text,
        compression=compression,
        processes=processes,
        reprocess=reprocess,
        trust_remote_code=trust_remote_code,
        args_str=str(args),
    )

log = logging.getLogger(__name__)

DONE_FILENAME = '.text_to_mds_conversion_done'


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert text files into MDS format, optionally concatenating and tokenizing',
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
        help='The folder to write output to',
    )
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='The wiki file which lines are sequences to convert to mds',
    )
    parser.add_argument(
        '--compression',
        type=str,
#        default='zstd',
        default=None,
        required=False,
        help='The compression algorithm to use for MDS writing',
    )

    #parser.add_argument(
    #    '--concat_tokens',
    #    type=int,
    #    required=True,
    #    help='Convert text to tokens and concatenate up to this many tokens',
    #)

    parser.add_argument(
        '--tokenizer',
        type=str,
        required=True,
        help='The name of the tokenizer to use',
    )
    parser.add_argument(
        '--bos_text',
        type=str,
        required=False,
        default=None,
        help=
        'The text to prepend to each example to separate concatenated examples',
    )
    parser.add_argument(
        '--eos_text',
        type=str,
        required=False,
        default=None,
        help=
        'The text to append to each example to separate concatenated examples',
    )
    parser.add_argument(
        '--use_tokenizer_eos',
        required=False,
        action='store_true',
        default=False,
        help='Use the EOS text from the tokenizer.',
    )
    #parser.add_argument(
    #    '--no_wrap',
    #    default=False,
    #    action='store_true',
    #    help=
    #    'Whether to let text examples wrap across multiple training examples',
    #)
    parser.add_argument(
        '--processes',
        type=int,
        required=False,
        default=min(max(psutil.cpu_count() - 2, 1), 32),
        help=
        'The number of processes to use to download and convert the dataset',
    )
    parser.add_argument(
        '--reprocess',
        type=bool,
        required=False,
        default=False,
        help='If true, reprocess the input_folder to mds format. Otherwise, ' +
        'only reprocess upon changes to the input folder or dataset creation parameters.',
    )
    parser.add_argument(
        '--trust-remote-code',
        type=bool,
        required=False,
        default=False,
        help='If true, allows custom code to be executed to load the tokenizer',
    )
    parser.add_argument(
        '--logging-level',
        type=str,
        required=False,
        default='INFO',
        help='Logging level for the script. Default is INFO.',
    )
    parsed = parser.parse_args()
    return parsed


if __name__ == '__main__':
    args = parse_args()
    convert_text_to_mds_from_args(
        output_folder=args.output_folder,
        input_file=args.input_file,
        compression=args.compression,
        tokenizer_name=args.tokenizer,
        bos_text=args.bos_text,
        eos_text=args.eos_text,
        use_tokenizer_eos=args.use_tokenizer_eos,
        processes=args.processes,
        reprocess=args.reprocess,
        trust_remote_code=args.trust_remote_code,
        logging_level=args.logging_level,
    )