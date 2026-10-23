"""Настройки приложения: адрес базы, сроки кэша и хранения, лимиты нагрузки,
административный токен. Читаются из .env, значения по умолчанию здесь же.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://postgres@localhost:5432/pricing"
    default_risk_free_rate: float = 0.042
    market_cache_ttl_minutes: int = 15

    rate_limit: str = "120/minute"

    # Монте-Карло гоняет симуляцию двенадцать раз: на цену и одиннадцать греков.
    pricing_rate_limit: str = "30/minute"

    # Потолок памяти: без него FastAPI отдал бы под расчёты сорок потоков.
    max_concurrent_pricing: int = 4

    # Пустой — сгенерировать случайный при старте, см. app/security.py.
    admin_token: str = ""

    calculation_retention_days: int = 3

    # Ниже двадцати улыбку волатильности строить не по чему.
    min_quoted_strikes: int = 20


settings = Settings()
