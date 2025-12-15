from collections import deque

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer


def tokenizing_distributed_data_loader_with_state(
    B,
    T,
    split,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
    device="cuda",
    resume_state_dict=None,
):
    """
    Document-aligned loader:
    - One document per batch (B must be 1)
    - No truncation allowed
    - If a document exceeds T+1 tokens, print raw text and raise an error
    """

    assert split in ["train", "val"]
    assert B == 1, "This loader supports only one document per batch (B=1)."

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    def document_batches():
        parquet_paths = list_parquet_files()
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict else None

        pq_idx = resume_pq_idx

        while True:
            while pq_idx < len(parquet_paths):
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)

                if resume_rg_idx is not None:
                    base_idx = resume_rg_idx // ddp_world_size
                    base_idx += 1
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    resume_rg_idx = None
                else:
                    rg_idx = ddp_rank

                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column("text").to_pylist()

                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i : i + tokenizer_batch_size], (pq_idx, rg_idx)

                    rg_idx += ddp_world_size

                pq_idx += 1

    batches = document_batches()

    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()

    def tokenized_documents():
        for doc_batch, (pq_idx, rg_idx) in batches:
            token_lists = tokenizer.encode(
                doc_batch, prepend=bos_token, num_threads=tokenizer_threads
            )
            for text, tokens in zip(doc_batch, token_lists):
                yield tokens, text, pq_idx, rg_idx

    doc_tokens_iter = tokenized_documents()

    max_allowed_len = T + 1
    use_cuda = device == "cuda"

    while True:
        tokens, text, pq_idx, rg_idx = next(doc_tokens_iter)

        if len(tokens) < 2:
            continue

        if len(tokens) > max_allowed_len:
            preview_len = 500
            preview = text[:preview_len].replace("\n", "\\n")
            print(
                "\n[DataLoader Error] Document exceeds block size.",
                f"\n  Token length: {len(tokens)} (> {max_allowed_len})",
                f"\n  Location: pq_idx={pq_idx}, rg_idx={rg_idx}",
                f"\n  Raw text (first {preview_len} chars):\n    \"{preview}\"",
                "\n",
                flush=True,
            )
            raise RuntimeError(
                f"Document token length {len(tokens)} exceeds T+1={max_allowed_len}. "
                f"Increase block size T or shorten the document."
            )

        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda)

        inputs_cpu = scratch[:-1].view(1, -1)
        targets_cpu = scratch[1:].view(1, -1)

        inputs = inputs_cpu.to(device=device, non_blocking=use_cuda)
        targets = targets_cpu.to(device=device, non_blocking=use_cuda)

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx}

        yield inputs, targets, state_dict


def tokenizing_distributed_data_loader(*args, **kwargs):
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(
        *args, **kwargs
    ):
        yield inputs, targets
