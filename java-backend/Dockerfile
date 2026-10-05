FROM node:22-alpine AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM gradle:8.8-jdk21 AS build
WORKDIR /workspace
COPY build.gradle settings.gradle ./
COPY java-backend ./java-backend
COPY --from=frontend /frontend/dist ./frontend/dist
RUN gradle --no-daemon :java-backend:bootJar

FROM eclipse-temurin:21-jre
WORKDIR /app
RUN useradd --system --uid 10001 appuser
COPY --from=build /workspace/java-backend/build/libs/java-backend-0.1.0.jar /app/app.jar
USER 10001
EXPOSE 8080
ENTRYPOINT ["java", "-XX:MaxRAMPercentage=75.0", "-jar", "/app/app.jar"]
