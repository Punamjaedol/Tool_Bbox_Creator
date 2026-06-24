CREATE TABLE `db_class_info` (
  `CLASS_ID` int NOT NULL AUTO_INCREMENT COMMENT '클래스 고유 번호 (자동 증가)',
  `CLASS_NAME` varchar(100) NOT NULL COMMENT '클래스명',
  `USE_YN` char(1) NOT NULL DEFAULT 'Y' COMMENT '사용 여부 (Y: 사용중, N: 삭제/숨김)',
  `REMARK` varchar(300) DEFAULT NULL COMMENT '비고',
  PRIMARY KEY (`CLASS_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='클래스 마스터 테이블';

CREATE TABLE `db_bbox_data` (
  `IMAGE_ID` bigint NOT NULL COMMENT '연관된 이미지 ID',
  `BBOX_SEQ` int NOT NULL COMMENT '이미지별 바운딩박스 순번',
  `CLASS_ID` int NOT NULL COMMENT '클래스 고유 번호 (db_class_info CLASS_ID 참조)',
  `X_MIN` float NOT NULL COMMENT '바운딩박스 좌상단 X 좌표',
  `Y_MIN` float NOT NULL COMMENT '바운딩박스 좌상단 Y 좌표',
  `X_MAX` float NOT NULL COMMENT '바운딩박스 우하단 X 좌표',
  `Y_MAX` float NOT NULL COMMENT '바운딩박스 우하단 Y 좌표',
  `CONFIDENCE` float DEFAULT NULL COMMENT 'AI 모델 신뢰도 (0.0 ~ 1.0)',
  `CREATE_TIME` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '라벨링 생성 일시',
  PRIMARY KEY (`IMAGE_ID`, `BBOX_SEQ`),
  KEY `idx_class_id` (`CLASS_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='이미지별 바운딩박스 복합키 라벨링 데이터';

select * from db_class_info