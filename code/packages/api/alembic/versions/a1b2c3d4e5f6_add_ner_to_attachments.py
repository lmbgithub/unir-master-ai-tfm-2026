"""add ner to attachments

Revision ID: a1b2c3d4e5f6
Revises: 576594dca4e5
Create Date: 2026-05-19 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "576594dca4e5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("attachments", sa.Column("ner", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("attachments", "ner")
