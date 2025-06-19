from datetime import datetime
from typing import Any

from sqlalchemy import String, JSON, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .database import Base


class ChatHistory(Base):
    __tablename__ = "chat_history"

    user_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    onboarding_history: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False
    )
    anc_survey_history: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), onupdate=func.now(), nullable=True
    )

    def __repr__(self):
        return (
            f"<ChatHistory(user_id='{self.user_id}', updated_at='{self.updated_at}')>"
        )
