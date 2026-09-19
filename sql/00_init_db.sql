-- Инициализация БД проекта. Выполняется один раз от суперпользователя postgres.
--
--   psql -U postgres -h localhost -d postgres -f sql/00_init_db.sql
--
-- Существующие базы на этом сервере не затрагиваются: создаётся отдельная
-- база pricing, приложение подключается к ней под пользователем postgres.

CREATE DATABASE pricing ENCODING 'UTF8';
