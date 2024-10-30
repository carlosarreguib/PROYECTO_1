-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema viviendas_bcn
-- -----------------------------------------------------
DROP SCHEMA IF EXISTS `viviendas_bcn`;
-- -----------------------------------------------------
-- Schema viviendas_bcn
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `viviendas_bcn` DEFAULT CHARACTER SET utf8 ;
USE `viviendas_bcn` ;

-- -----------------------------------------------------
-- Table `viviendas_bcn`.`kaggle`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `viviendas_bcn`.`kaggle`;
CREATE TABLE IF NOT EXISTS `viviendas_bcn`.`kaggle` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `price` INT NULL,
  `rooms` INT NULL,
  `bathroom` INT NULL,
  `lift` VARCHAR(10) NULL,
  `terrace` VARCHAR(10) NULL,
  `square_meters` INT NULL,
  `real_state` VARCHAR(45) NULL,
  `neighborhood` VARCHAR(45) NULL,
  `square_meters_price` FLOAT NULL,
  PRIMARY KEY (`id`))
ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
