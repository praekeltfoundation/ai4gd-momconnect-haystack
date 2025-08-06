from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base

# TODO: Add a check to the github actions for schema changes without migrations


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
        DateTime(timezone=True), default=datetime.now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), onupdate=datetime.now, nullable=True
    )

    __table_args__ = {"extend_existing": True}

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
        DateTime(timezone=True), default=datetime.now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), onupdate=datetime.now, nullable=True
    )

    __table_args__ = {"extend_existing": True}

    def __repr__(self):
        return f"<AssessmentHistory(user_id='{self.user_id}', updated_at='{self.updated_at}')>, question_number='{self.question_number}', score='{self.score}'"


class AssessmentResultHistory(Base):
    __tablename__ = "assessment_result_history"

    user_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    assessment_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    category: Mapped[str] = mapped_column(String, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    crossed_skip_threshold: Mapped[bool] = mapped_column(Boolean, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), onupdate=datetime.now, nullable=True
    )

    __table_args__ = {"extend_existing": True}

    def __repr__(self):
        return f"<AssessmentResultHistory(user_id='{self.user_id}', updated_at='{self.updated_at}', category='{self.category}', crossed_skip_threshold='{self.crossed_skip_threshold}')>"


class AssessmentEndMessagingHistory(Base):
    __tablename__ = "assessment_end_messaging_history"

    user_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    assessment_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    message_number: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_response: Mapped[str] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), onupdate=datetime.now, nullable=True
    )

    __table_args__ = {"extend_existing": True}

    def __repr__(self):
        return f"<AssessmentEndMessagingHistory(user_id='{self.user_id}', updated_at='{self.updated_at}')>"


class UserJourneyState(Base):
    __tablename__ = "user_journey_state"

    user_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    current_flow_id: Mapped[str] = mapped_column(String, nullable=False)
    current_step_identifier: Mapped[str] = mapped_column(String, nullable=False)
    last_question_sent: Mapped[str] = mapped_column(String, nullable=False)
    user_context: Mapped[dict] = mapped_column(JSON, nullable=True)
    reminder_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), onupdate=datetime.now, default=datetime.now
    )

    __table_args__ = {"extend_existing": True}

    def __repr__(self):
        return f"<UserJourneyState(user_id='{self.user_id}', flow_id='{self.current_flow_id}', step='{self.current_step_identifier}')>"
