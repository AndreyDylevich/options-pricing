"""Замена справочника бумаг целиком.

    python -m scripts.sync_instruments --replace

Живёт отдельной командой, а не флагом эндпоинта, намеренно. Замена
удаляет из `instruments` всё, чего нет в новом ответе скринера, а у
`market_snapshots` внешний ключ на эту таблицу с ON DELETE CASCADE —
значит вместе со справочником уходит и кэш рыночных снимков. Такому не
место в HTTP-ручке: сервер слушает localhost, но послать на него POST
может любая страница, открытая в браузере пользователя.

Без `--replace` делает то же, что кнопка на странице: дополняет и
освежает, ничего не удаляя.
"""

import argparse
import asyncio
import sys

from sqlalchemy import select

from app.db import repository
from app.db.entities import Instrument
from app.db.session import SessionFactory, engine
from app.services import market_data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--replace",
        action="store_true",
        help="удалить бумаги, которых нет в новом ответе скринера",
    )
    parser.add_argument("--screener", default=market_data.DEFAULT_SCREENER)
    parser.add_argument(
        "--count", type=int, default=market_data.DEFAULT_SCREENER_COUNT
    )
    parser.add_argument(
        "--yes", action="store_true", help="не спрашивать подтверждения"
    )
    return parser.parse_args(argv)


# Показать, что будет удалено, и дождаться внятного согласия. Список
# печатается целиком: увидеть заранее, что скринер сегодня не отдал половину
# привычных бумаг, гораздо дешевле, чем восстанавливать справочник вместе с
# кэшем снимков.
def confirm(doomed: list[str]) -> bool:
    print(f"\nБудет удалено бумаг: {len(doomed)}")
    print("  " + ", ".join(doomed))
    print("Вместе с ними каскадом уйдут их рыночные снимки.\n")

    return input("Введите «да» для продолжения: ").strip().lower() == "да"


async def run(args: argparse.Namespace) -> int:
    print(f"Спрашиваем скринер {args.screener}, бумаг {args.count}…")
    items = await asyncio.to_thread(
        market_data.fetch_screener_instruments, args.screener, args.count
    )
    print(f"Скринер отдал: {len(items)}")

    async with SessionFactory() as session:
        if args.replace:
            known = set(
                (await session.scalars(select(Instrument.ticker))).all()
            )
            doomed = sorted(known - {item.ticker for item in items})

            if doomed and not args.yes and not confirm(doomed):
                print("Отменено, ничего не изменилось.")
                return 1

        summary = await repository.sync_instruments(
            session, items, replace=args.replace
        )

    print(
        f"Готово: новых {summary['added']}, обновлено {summary['updated']}, "
        f"удалено {summary.get('removed', 0)}"
    )
    return 0


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(run(args))
    finally:
        asyncio.run(engine.dispose())


if __name__ == "__main__":
    sys.exit(main())
