from datetime import datetime

from sqlalchemy import Integer, String, JSON, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .database import Base


class ChatHistory(Base):
    __tablename__ = "chat_history"

    user_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    onboarding_history: Mapped[list[dict[str, str]]] = mapped_column(
        JSON, nullable=False
    )
    anc_survey_history: Mapped[list[dict[str, str]]] = mapped_column(
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


class AssessmentHistory(Base):
    __tablename__ = "assessment_history"

    user_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    assessment_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    question_number: Mapped[int] = mapped_column(primary_key=True, index=True)
    question: Mapped[str] = mapped_column(String, nullable=False)
    user_response: Mapped[str] = mapped_column(String, nullable=True)
    score: Mapped[int] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), onupdate=func.now(), nullable=True
    )

    def __repr__(self):
        return (
            f"<AssessmentHistory(user_id='{self.user_id}', updated_at='{self.updated_at}')>"
        )


class AssessmentResultHistory(Base):
    __tablename__ = "assessment_result_history"

    user_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    assessment_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    category: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), onupdate=func.now(), nullable=True
    )

    def __repr__(self):
        return (
            f"<AssessmentResultHistory(user_id='{self.user_id}', updated_at='{self.updated_at}')>"
        )


class AssessmentEndMessagingHistory(Base):
    __tablename__ = "assessment_end_messaging_history"

    user_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    assessment_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    message_number: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_response: Mapped[str] = mapped_column(String, nullable=True)

    def __repr__(self):
        return (
            f"<AssessmentEndMessagingHistory(user_id='{self.user_id}', updated_at='{self.updated_at}')>"
        )
