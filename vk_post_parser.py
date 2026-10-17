"""Сбор постов сообщества VK за заданный период и сохранение в CSV"""

import os
import argparse
import csv
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import TypedDict
from zoneinfo import ZoneInfo

import vk_api
from dotenv import load_dotenv
from vk_api.exceptions import ApiError
from vk_api.vk_api import VkApiMethod

load_dotenv()

BATCH_SIZE = 100


class Post(TypedDict):
    """Поля собранного поста для сохранения в CSV"""
    id: int | None  # Идентификатор поста
    owner_id: int | None  # Идентификатор владельца стены
    date: str  # Дата публикации в формате
    likes: int  # Количество отметок «Нравится»
    comments: int  # Количество комментариев
    reposts: int  # Количество репостов
    views: int  # Количество просмотров
    url: str  # Ссылка на пост в VK
    is_repost: bool  # Является ли запись репостом (есть copy_history)
    text: str  # Собственный текст и текст исходных записей без переносов строк


def parse_args() -> argparse.Namespace:
    """Получить аргументы командной строки"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-gd", "--group_domain", default="samara_university", help="Домен группы")
    parser.add_argument("-sd", "--start_date", default="2026-08-01", help="Дата начала")
    parser.add_argument("-ed", "--end_date", default="2026-08-31", help="Дата конца")
    parser.add_argument("-o", "--output", default="output/posts.csv", help="Вывод")
    return parser.parse_args()


def create_vk_api(vk_token: str = "VK_TOKEN") -> VkApiMethod:
    """Создать клиент VK API с токеном из переменной окружения"""

    token = os.getenv(vk_token)
    if not token:
        raise RuntimeError("Токен не найден. Добавьте его в файл .env.")

    session = vk_api.VkApi(token=token)
    return session.get_api()


def get_posts(
    vk: VkApiMethod,
    domain: str,
    start_date: str,
    end_date: str,
) -> tuple[list[Post], bool]:
    """Вернуть посты и статус сбора за период, включая граничные даты"""

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1))

    posts: list[Post] = []
    is_complete = True

    offset = 0
    seen_posts = set()

    print(f"Получение постов сообщества {domain} с {start_date} по {end_date}")

    while True:
        print(f"Загрузка записей: offset={offset}")

        try:
            response = vk.wall.get(domain=domain, count=BATCH_SIZE, offset=offset)
        except ApiError as error:
            is_complete = False
            print("Ошибка VK API:")
            print(error)
            break

        items = response.get("items", [])

        # Если записей больше нет
        if not items:
            break

        for post in items:
            post_id = post.get("id")
            owner_id = post.get("owner_id")

            # Защита от возможных повторов
            unique_id = (owner_id, post_id)
            if unique_id in seen_posts:
                continue

            seen_posts.add(unique_id)

            timestamp = post.get("date")
            if timestamp is None:
                continue

            post_datetime = datetime.fromtimestamp(timestamp)

            # Пост попадает в заданный диапазон
            if start_dt <= post_datetime < end_dt:
                copy_history = post.get("copy_history", [])
                # Сохраняем комментарий к репосту и тексты исходных записей.
                text_parts = [post.get("text", "")]
                text_parts.extend(original.get("text", "") for original in copy_history)
                text = "\n".join(part for part in text_parts if part.strip())

                post_data: Post = {
                    "id": post_id,
                    "owner_id": owner_id,
                    "date": post_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                    "likes": post.get("likes", {}).get("count", 0),
                    "comments": post.get("comments", {}).get("count", 0),
                    "reposts": post.get("reposts", {}).get("count", 0),
                    "views": post.get("views", {}).get("count", 0),
                    "url": (f"https://vk.ru/wall{owner_id}_{post_id}"),
                    "is_repost": bool(copy_history),
                    "text": text.replace("\n", " "),
                }
                posts.append(post_data)
                print(f"  + {post_datetime:%Y-%m-%d %H:%M} id={post_id}")

        # Закреплённая запись может нарушать хронологический порядок, поэтому её здесь не учитываем
        normal_posts = [post for post in items if not post.get("is_pinned")]

        if normal_posts:
            newest_normal_post = max(datetime.fromtimestamp(post["date"]) for post in normal_posts)

            # Если даже самая новая обычная запись на странице уже старше интересующего периода, дальше можно не загружать
            if newest_normal_post < start_dt:
                print(f"Достигнуты записи старше {start_date}. Останавливаем загрузку.")
                break

        offset += len(items)
        # Дополнительная защита, VK сообщил, что дошли до конца стены
        if offset >= response.get("count", 0):
            break

    # Сортировка от старых постов к новым
    posts.sort(key=lambda post: post["date"])
    return posts, is_complete


def save_to_csv(
    posts: Sequence[Mapping[str, object]],
    filename: str | os.PathLike[str],
) -> None:
    """Сохранить посты в CSV, создав родительские каталоги при необходимости"""

    if not posts:
        print("Нет постов для сохранения.")
        return

    fieldnames = list(dict.fromkeys(key for post in posts for key in post))
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(posts)

    print()
    print(f"CSV сохранён: {filename}")
    print(f"Количество постов: {len(posts)}")


def main() -> int:
    """Выгрузить посты и вернуть код 0 при полном сборе или 1 при ошибке API"""
    args = parse_args()
    vk = create_vk_api()
    posts, is_complete = get_posts(
        vk=vk,
        domain=args.group_domain,
        start_date=args.start_date,
        end_date=args.end_date
    )
    output = Path(args.output)
    if not is_complete:
        output = output.with_name(f"{output.stem}.partial{output.suffix}")
        print("ВНИМАНИЕ: выгрузка неполная из-за ошибки VK API.")

    save_to_csv(posts, output)
    return 0 if is_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
